from abc import ABC, abstractmethod


class Probe(ABC):
    """Record one timestep of specified data."""

    attribute: str

    @abstractmethod
    def process(self, neurons):
        pass


class GeneralProbe(Probe):
    """Generic probe for recording one attribute from neurons."""

    def __init__(self, attribute: str):
        self.attribute = attribute

    def process(self, neurons):
        return getattr(neurons, self.attribute)


class VoltageProbe(GeneralProbe):
    """Record voltage of all neurons at time t."""

    def __init__(self, attribute: str = "voltage"):
        super().__init__(attribute)


class SpikeProbe(GeneralProbe):
    """Record spike of all neurons at time t."""

    def __init__(self, attribute: str = "spike"):
        super().__init__(attribute)


class CurrentProbe(GeneralProbe):
    """Record current of all neurons at time t."""

    def __init__(self, attribute: str = "current"):
        super().__init__(attribute)


class ActivationProbe(GeneralProbe):
    """Record activation of all neurons at time t."""

    def __init__(self, attribute: str = "activation"):
        super().__init__(attribute)


class Probes:
    def __init__(self, probes: list[Probe] | None = None):
        if probes is None:
            probes = []
        self.probes = probes

    def process(self, neurons):
        return {probe.attribute: probe.process(neurons) for probe in self.probes}
