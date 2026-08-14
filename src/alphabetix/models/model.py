from ..module import Module
from .decoder_model import DecoderModel
from .input_model import InputModel
from .network_model import NetworkModel
from .neuron_model import NeuronModel


class Model(Module):
    decoder_model: DecoderModel
    input_model: InputModel
    network_model: NetworkModel
    neuron_model: NeuronModel
    dt: float = Module.static()

    def __init__(
        self,
        decoder_model: DecoderModel,
        input_model: InputModel,
        network_model: NetworkModel,
        neuron_model: NeuronModel,
        dt: float,
    ):
        self.decoder_model = decoder_model
        self.input_model = input_model
        self.network_model = network_model
        self.neuron_model = neuron_model
        self.dt = dt
