from abc import abstractmethod

from ..module import Module


class NeuronModel(Module):
    @abstractmethod
    def update(
        self,
        neuron,  # the state
        activation,
        current,
        utilization,
        resource,
        dt,
    ):
        pass
