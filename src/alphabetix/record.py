from abc import ABC, abstractmethod


class Probe(ABC):
    """Record one timestep of specified data."""

    attribute: str

    @abstractmethod
    def process(self, neurons):
        pass


class GenericProbe(Probe):
    """Generic probe for recording one attribute from neurons."""

    def __init__(self, attribute: str):
        self.attribute = attribute

    def process(self, neurons):
        return getattr(neurons, self.attribute)


class VoltageProbe(GenericProbe):
    """Record voltage of all neurons at time t."""

    def __init__(self, attribute: str = "voltage"):
        super().__init__(attribute)


class SpikeProbe(GenericProbe):
    """Record spike of all neurons at time t."""

    def __init__(self, attribute: str = "spike"):
        super().__init__(attribute)


class CurrentProbe(GenericProbe):
    """Record total current of all neurons at time t."""

    def __init__(self, attribute: str = "current"):
        super().__init__(attribute)


class SynapticCurrentProbe(GenericProbe):
    """Record synaptic current of all neurons at time t."""

    def __init__(self, attribute: str = "synaptic_current"):
        super().__init__(attribute)


class BackgroundCurrentProbe(GenericProbe):
    """Record cortical background current of all neurons at time t."""

    def __init__(self, attribute: str = "background_current"):
        super().__init__(attribute)


class TaskInputCurrentProbe(GenericProbe):
    """Record task-driven input current of all neurons at time t."""

    def __init__(self, attribute: str = "task_input_current"):
        super().__init__(attribute)


class ActivationProbe(GenericProbe):
    """Record activation of all neurons at time t."""

    def __init__(self, attribute: str = "activation"):
        super().__init__(attribute)


class RefractoryTimeProbe(GenericProbe):
    """Record activation of all neurons at time t."""

    def __init__(self, attribute: str = "refractory_time_remaining"):
        super().__init__(attribute)


class Probes:
    def __init__(self, probes: list[Probe] | None = None):
        if probes is None:
            probes = []
        self.probes = probes

    def process(self, neurons):
        return {probe.attribute: probe.process(neurons) for probe in self.probes}
