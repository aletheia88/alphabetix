import jax

from ..module import Module


class Network(Module):
    synapse_taus: jax.Array  # (num_neurons, num_neurons)
    synapse_activations: jax.Array  # (num_neurons, num_neurons)
