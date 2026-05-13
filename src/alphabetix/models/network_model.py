import jax

from ..module import Module
from .neuron_model import NeuronModel, Constants


class NetworkModel(Module):
    # rows: post-synaptic neurons
    # columns: pre-synaptic neurons
    # connectivity: conductance weights, unit nS
    connectivity: jax.Array  # (num_neurons, num_neurons)

    def step(self, neurons, input_activations, dt):
        activations, currents = self._compute_activations_and_currents(
            neurons, input_activations, dt
        )
        next_neurons = jax.vmap(NeuronModel.update, in_axes=(0, 0, 0, None))(
            neurons, activations, currents, dt
        )
        return next_neurons

    def _compute_activations_and_currents(self, neurons, input_activations, dt):
        # update activations according to connectivity
        # TODO: tau should be synaptic decay time constant (neuron-type
        # specific) not the generic membrane decay time constant
        exc_presynaptic_neurons = neurons.sign > 0
        inh_presynaptic_neurons = neurons.sign < 0

        network_activations = (
            1 - dt / neurons.tau_synapse
        ) * neurons.activation + self.connectivity @ (neurons.spike * neurons.sign)

        exc_connectivity = self.connectivity * exc_presynaptic_neurons[None, :]
        inh_connectivity = self.connectivity * inh_presynaptic_neurons[None, :]

        exc_current = (
            exc_connectivity
            @ (network_activations + input_activations)
            * (neurons.voltage - Constants.exc_reversal_potential)
        )
        inh_current = (
            inh_connectivity
            @ network_activations
            * (neurons.voltage - Constants.inh_reversal_potential)
        )

        return network_activations, exc_current + inh_current
