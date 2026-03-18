from abc import abstractmethod

import equinox as eqx

from .state import NeuronStates


class Sensor(eqx.Module):
    @abstractmethod
    def measure(self, states: NeuronStates):
        pass
