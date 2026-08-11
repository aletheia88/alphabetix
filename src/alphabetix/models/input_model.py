from abc import abstractmethod

import jax
import jax.numpy as jnp

from ..module import Module
from .sensory_model import SensoryModel
from .timeline import TimelineInputs
from .topdown_model import TopDownModel


@abstractmethod
class InputModel(Module):
    """Abstract base class for all input-current generators."""

    @abstractmethod
    def compute_currents(self, timeline_inputs: TimelineInputs | None) -> jax.Array:
        """Return input currents with shape (num_timesteps, num_neurons)."""
        raise NotImplementedError


class ExplicitInputModel(InputModel):
    currents: jax.Array = Module.param()

    def compute_currents(self, timeline_inputs=None):
        return self.currents


class TimelineInputModel(InputModel):
    # component models
    sensory_model: SensoryModel
    topdown_model: TopDownModel

    num_cues: int = Module.static()
    num_categories: int = Module.static()
    num_exc_neurons: int = Module.static()
    num_inh_neurons: int = Module.static()
    num_neurons: int = Module.static()

    def __init__(
        self,
        num_cues: int,
        num_categories: int,
        num_exc_neurons: int,
        num_inh_neurons: int,
        percent_coding: float,
        percent_unique_coding: float,
        percent_shared_coding: float,
        stim_amplitude: float,
        *,
        key: jax.Array,
    ):
        self.num_cues = num_cues
        self.num_categories = num_categories

        self.num_exc_neurons = num_exc_neurons
        self.num_inh_neurons = num_inh_neurons
        self.num_neurons = num_exc_neurons + num_inh_neurons

        key1, key2 = jax.random.split(key)

        self.sensory_model = SensoryModel(
            num_categories,
            num_exc_neurons,
            percent_coding,
            percent_unique_coding,
            percent_shared_coding,
            stim_amplitude,
            key=key1,
        )
        self.topdown_model = TopDownModel(self.num_cues, num_inh_neurons, key=key2)

    def compute_currents(self, timeline_inputs):
        if timeline_inputs is None:
            raise ValueError(
                "TimelineInputModel requires TimelineInputs. Pass "
                "`timeline.get_inputs(dt)` to `run_simulation` or `train_step`."
            )
        # collect encoding inputs over all timesteps
        temporal = timeline_inputs.temporal_encodings
        category = timeline_inputs.category_encodings

        sensory_inputs = jax.vmap(self.sensory_model)(category)
        topdown_inputs = jax.vmap(self.topdown_model)(temporal)
        # NOTE: as only exc, inh receive sensory, topdown respectively, we can do this:
        # [excitatory neurons | inhibitory neurons]
        # [ sensory input     | top-down input      ]
        return jnp.concatenate(
            (sensory_inputs, topdown_inputs),
            axis=-1,
        )
