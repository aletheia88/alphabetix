from abc import abstractmethod

from ..module import Module


class NeuronModel(Module):
    @abstractmethod
    def update(self, neuron, dt):
        pass
