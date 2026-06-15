import jax
import jax.numpy as jnp

from ..module import Module
from .neuron_model import NeuronModel, Constants


class NetworkModel(Module):
    # rows: post-synaptic neurons
    # columns: pre-synaptic neurons
    # connectivity: conductance weights, unit nS
    connectivity: jax.Array  # (num_neurons, num_neurons)
    synapse_taus: jax.Array  # (num_neurons, num_neurons)
    synapse_activations: jax.Array  # (num_neurons, num_neurons)
    spike_buffer: jax.Array  # (delay_steps, num_neurons)

    def step(self, neurons, inputs, dt):
        (
            next_network,
            activations,
            internal_exc_currents,
            input_exc_currents,
            exc_currents,
            inh_currents,
            utilization,
            resource,
        ) = self._compute_activations_and_currents(neurons, inputs, dt)
        next_neurons = jax.vmap(
            NeuronModel.update, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            neurons,
            activations,
            internal_exc_currents,
            input_exc_currents,
            exc_currents,
            inh_currents,
            utilization,
            resource,
            dt,
        )
        return next_network, next_neurons

    def _compute_activations_and_currents(self, neurons, input_activations, dt):
        exc_mask = neurons.sign > 0
        inh_mask = neurons.sign < 0
        ee_mask = exc_mask[:, None] & exc_mask[None, :]

        last_u = neurons.utilization
        last_x = neurons.resource

        arriving_spikes, next_spike_buffer = self._get_delayed_spikes_and_update_buffer(
            neurons.spike
        )

        plain_increment = self.connectivity * arriving_spikes
        stp_increment = plain_increment * (last_u * last_x)
        # short-term plasticity only applies to exc-exc connections
        increment = jnp.where(ee_mask, stp_increment, plain_increment)

        next_synapse_activations = (
            1 - dt / self.synapse_taus
        ) * self.synapse_activations + increment

        exc_synapse_activations = next_synapse_activations * exc_mask
        inh_synapse_activations = next_synapse_activations * inh_mask

        # update STP variables
        fired_exc = arriving_spikes * exc_mask

        u_temp = last_u + dt * (Constants.u_total - last_u) / Constants.tau_f
        u = u_temp + 0.75 * (1 - u_temp) * fired_exc
        x_temp = last_x + dt * (1 - last_x) / Constants.tau_d
        x = x_temp - u_temp * x_temp * fired_exc

        exc_activations = jnp.sum(exc_synapse_activations, axis=1)
        inh_activations = jnp.sum(inh_synapse_activations, axis=1)

        activations = exc_activations + inh_activations
        # exc_currents = (exc_activations + input_activations) * (
        #     neurons.voltage - Constants.exc_reversal_potential
        # )
        internal_exc_currents = exc_activations * (
            neurons.voltage - Constants.exc_reversal_potential
        )
        input_exc_currents = input_activations * (
            neurons.voltage - Constants.exc_reversal_potential
        )
        exc_currents = internal_exc_currents + input_exc_currents

        inh_currents = (inh_activations) * (
            neurons.voltage - Constants.inh_reversal_potential
        )
        # currents = exc_currents + inh_currents

        next_network = self.replace(
            synapse_activations=next_synapse_activations,
            spike_buffer=next_spike_buffer,
        )

        return (
            next_network,
            activations,
            internal_exc_currents,
            input_exc_currents,
            exc_currents,
            inh_currents,
            u,
            x,
        )

    def _get_delayed_spikes_and_update_buffer(self, current_spikes):
        arriving_spikes = self.spike_buffer[0]
        # update spike buffer
        next_spike_buffer = jnp.concatenate(
            [
                self.spike_buffer[1:],
                current_spikes[None, :],
            ],
            axis=0,
        )
        return arriving_spikes, next_spike_buffer
