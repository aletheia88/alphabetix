from abc import abstractmethod

import equinox as eqx


class Sensor(eqx.Module):
    @abstractmethod
    def measure(self):
        pass
