import jax

from ..module import Module
from .neuron_model import NeuronModel, Constants


class NetworkModel(Module):
    # rows: post-synaptic neurons
    # columns: pre-synaptic neurons
    # connectivity: conductance weights, unit nS
    connectivity: jax.Array  # (num_neurons, num_neurons)

    def step(self, neurons, input_activations, dt):
        exc_activations, inh_activations, exc_currents, inh_currents = (
            self._compute_activations_and_currents(neurons, input_activations, dt)
        )
        next_neurons = jax.vmap(NeuronModel.update, in_axes=(0, 0, 0, 0, 0, None))(
            neurons, exc_activations, inh_activations, exc_currents, inh_currents, dt
        )
        return next_neurons

    def _compute_activations_and_currents(self, neurons, input_activations, dt):
        exc_presynaptic_neurons = neurons.sign > 0
        inh_presynaptic_neurons = neurons.sign < 0

        exc_connectivity = self.connectivity * exc_presynaptic_neurons[None, :]
        inh_connectivity = self.connectivity * inh_presynaptic_neurons[None, :]

        # activation sources are excitatory neurons
        exc_activations = (
            1 - dt / neurons.tau_synapse
        ) * neurons.exc_activation + exc_connectivity @ neurons.spike

        # activation sources are inhibitory neurons
        inh_activations = (
            1 - dt / neurons.tau_synapse
        ) * neurons.inh_activation + inh_connectivity @ neurons.spike

        # currents from (presynaptic) excitatory neurons
        exc_currents = (exc_activations + input_activations) * (
            neurons.voltage - Constants.exc_reversal_potential
        )
        # currents from (presynaptic) inhibitory neurons
        inh_currents = (inh_activations + input_activations) * (
            neurons.voltage - Constants.inh_reversal_potential
        )

        return (
            exc_activations,
            inh_activations,
            exc_currents,
            inh_currents,
        )
