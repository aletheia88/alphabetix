import jax
import jax.numpy as jnp

from ..module import Module
from .constants import Constants
from .neuron_model import NeuronModel


class NetworkModel(Module):
    # rows: post-synaptic neurons
    # columns: pre-synaptic neurons
    # connectivity: conductance weights, unit nS
    connectivity: jax.Array  # (num_neurons, num_neurons)
    synapse_taus: jax.Array  # (num_neurons, num_neurons)
    synapse_activations: jax.Array  # (num_neurons, num_neurons)

    def step(self, neurons, inputs, dt):
        (
            next_network,
            activations,
            currents,
            utilization,
            resource,
        ) = self._compute_activations_and_currents(neurons, inputs, dt)
        next_neurons = jax.vmap(NeuronModel.update, in_axes=(0, 0, 0, 0, 0, None))(
            neurons,
            activations,
            currents,
            utilization,
            resource,
            dt,
        )
        return next_network, next_neurons

    def _compute_activations_and_currents(self, neurons, input_currents, dt):
        exc_mask = neurons.sign > 0
        inh_mask = neurons.sign < 0
        ee_mask = exc_mask[:, None] & exc_mask[None, :]

        last_u = neurons.utilization
        last_x = neurons.resource

        # plain_increment = self.connectivity * arriving_spikes
        plain_increment = self.connectivity * neurons.spike
        stp_increment = plain_increment * (last_u * last_x)
        # short-term plasticity only applies to exc-exc connections
        increment = jnp.where(ee_mask, stp_increment, plain_increment)

        next_synapse_activations = (
            1 - dt / self.synapse_taus
        ) * self.synapse_activations + increment

        exc_synapse_activations = next_synapse_activations * exc_mask
        inh_synapse_activations = next_synapse_activations * inh_mask

        # update STP variables
        # fired_exc = arriving_spikes * exc_mask
        fired_exc = neurons.spike * exc_mask

        u_temp = last_u + dt * (Constants.u_total - last_u) / Constants.tau_f
        u = u_temp + 0.75 * (1 - u_temp) * fired_exc
        x_temp = last_x + dt * (1 - last_x) / Constants.tau_d
        x = x_temp - u_temp * x_temp * fired_exc

        exc_activations = jnp.sum(exc_synapse_activations, axis=1)
        inh_activations = jnp.sum(inh_synapse_activations, axis=1)

        activations = exc_activations + inh_activations

        exc_currents = exc_activations * (
            neurons.voltage - Constants.exc_reversal_potential
        )
        inh_currents = inh_activations * (
            neurons.voltage - Constants.inh_reversal_potential
        )
        currents = exc_currents + inh_currents + input_currents

        next_network = self.replace(
            synapse_activations=next_synapse_activations,
        )

        return (
            next_network,
            activations,
            currents,
            u,
            x,
        )
