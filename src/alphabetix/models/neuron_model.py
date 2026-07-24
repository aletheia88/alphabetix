from abc import abstractmethod

from ..module import Module


class Neuron(Module):
    pass


class NeuronModel(Module):
    @abstractmethod
    def update(
        self,
        neuron,  # the state
        activation,
        current,
        dt,
    ):
        pass
