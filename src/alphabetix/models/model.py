from ..module import Module
from .input_model import InputModel
from .network import Network
from .network_model import NetworkModel
from .neuron import Neuron


class Model(Module):
    input_model: InputModel
    network_model: NetworkModel
    initial_network: Network
    initial_neurons: Neuron
    dt: float = Module.static()

    def __init__(
        self,
        input_model,
        network_model,
        initial_network,
        initial_neurons,
        dt: float,
    ):
        self.input_model = input_model
        self.network_model = network_model
        self.initial_network = initial_network
        self.initial_neurons = initial_neurons
        self.dt = dt

    def partition(self):
        """Split into into trainable and non-trainable parameters with a custom mask."""
        return eqx.partition(self, self.trainable_filter())

    def trainable_filter(self):
        """Return a PyTree mask selecting only desired learnable leaves."""
        trainable = jax.tree.map(lambda _: False, self)

        input_filter = jax.tree.map(eqx.is_inexact_array, self.input_model)
        trainable = eqx.tree_at(
            lambda model: model.input_model,
            trainable,
            input_filter,
        )
        trainable = eqx.tree_at(
            lambda model: model.network_model.connectivity,
            trainable,
            True,
        )
        return trainable
