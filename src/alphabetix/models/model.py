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
