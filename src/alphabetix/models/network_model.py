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

    def step(self, neurons, input_activations, dt):
        next_network, activations, currents = self._compute_activations_and_currents(
            neurons, input_activations, dt
        )
        next_neurons = jax.vmap(NeuronModel.update, in_axes=(0, 0, 0, None))(
            neurons, activations, currents, dt
        )
        return next_network, next_neurons

    def _compute_activations_and_currents(self, neurons, input_activations, dt):
        next_synapse_activations = (
            1 - dt / self.synapse_taus
        ) * self.synapse_activations + self.connectivity * neurons.spike

        exc_synapse_activations = next_synapse_activations * (neurons.sign > 0)
        inh_synapse_activations = next_synapse_activations * (neurons.sign < 0)

        exc_activations = jnp.sum(exc_synapse_activations, axis=1)
        inh_activations = jnp.sum(inh_synapse_activations, axis=1)

        exc_currents = (exc_activations + input_activations) * (
            neurons.voltage - Constants.exc_reversal_potential
        )
        inh_currents = (inh_activations) * (
            neurons.voltage - Constants.inh_reversal_potential
        )
        currents = exc_currents + inh_currents

        activations = exc_activations + inh_activations

        next_network = self.replace(
            synapse_activations=next_synapse_activations,
        )

        return next_network, activations, currents
