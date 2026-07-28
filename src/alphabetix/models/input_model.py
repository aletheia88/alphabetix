from abc import abstractmethod

import jax
import jax.numpy as jnp

from ..module import Module
from .sensory_model import SensoryModel
from .timeline import Timeline
from .topdown_model import TopDownModel


@abstractmethod
class InputModel(Module):
    """Abstract base class for all input-current generators."""

    @abstractmethod
    def compute_currents(self, dt: float) -> jax.Array:
        """Return input currents with shape (num_timesteps, num_neurons)."""
        raise NotImplementedError


class ExplicitInputModel(InputModel):
    currents: jax.Array = Module.param()

    def compute_currents(self, dt: float):
        return self.currents


class TonicInputModel(InputModel):
    exc_bias: jax.Array = Module.param()
    inh_bias: jax.Array = Module.param()
    cue_amplitude: jax.Array = Module.param()

    num_exc: int = Module.static()
    num_inh: int = Module.static()
    num_timesteps: int = Module.static()
    cue_steps: int = Module.static()

    def compute_currents(self, dt):
        exc = jnp.full(
            (self.num_timesteps, self.num_exc),
            self.exc_bias,
        )
        inh = jnp.full(
            (self.num_timesteps, self.num_inh),
            self.inh_bias,
        )

        inputs = jnp.concatenate([exc, inh], axis=1)

        cue = jnp.arange(self.num_timesteps) < self.cue_steps
        inputs = inputs.at[:, : self.num_exc].add(cue[:, None] * self.cue_amplitude)
        return inputs


class TimelineInputModel(InputModel):
    # component models
    sensory_model: SensoryModel
    topdown_model: TopDownModel

    timeline: Timeline = Module.static()
    num_cues: int = Module.static()
    num_categories: int = Module.static()
    num_exc_neurons: int = Module.static()
    num_inh_neurons: int = Module.static()
    num_neurons: int = Module.static()

    def __init__(
        self,
        timeline: Timeline,
        num_exc_neurons: int,
        num_inh_neurons: int,
        *,
        key: jax.Array,
    ):
        self.timeline = timeline
        self.num_cues = timeline.num_cues
        self.num_categories = timeline.num_categories

        self.num_exc_neurons = num_exc_neurons
        self.num_inh_neurons = num_inh_neurons
        self.num_neurons = num_exc_neurons + num_inh_neurons

        key1, key2 = jax.random.split(key)

        self.sensory_model = SensoryModel(
            self.num_categories, self.num_neurons, key=key1
        )
        self.topdown_model = TopDownModel(self.num_cues, num_inh_neurons, key=key2)

    def compute_currents(self, dt: float):
        num_timesteps = int(self.timeline.total_time / dt)
        print(f"number of timesteps: {num_timesteps}")
        sensory_inputs = jnp.zeros((num_timesteps, self.num_neurons), dtype=jnp.float32)
        topdown_inputs = jnp.zeros((num_timesteps, self.num_neurons), dtype=jnp.float32)
        for i in range(num_timesteps):
            # convert the i-th step to time stamp in a trial
            t = int(i * dt)
            temporal_encoding = self.timeline.lookup_temporal(t)
            category_encoding = self.timeline.lookup_category(t)
            sensory_input = self.sensory_model(category_encoding)
            topdown_input = self.topdown_model(temporal_encoding)

            sensory_inputs = sensory_inputs.at[i, :].set(sensory_input)
            topdown_inputs = topdown_inputs.at[i, self.num_exc_neurons :].set(
                topdown_input
            )
        return sensory_inputs + topdown_inputs
