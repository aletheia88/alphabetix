import equinox as eqx
import jax
import jax.numpy as jnp

from .timeline import Timeline

from ..module import Module


class DecoderModel(Module):
    linear: eqx.nn.Linear
    spikes: jax.Array
    num_categories: int = eqx.field(static=True)

    # inputs to the decoder:
    # 1. rastor plot: (num_timesteps, num_neurons)
    # or time-windowed rastor plot: (last_delay_timesteps, num_neurons)
    # 2. prompt token: (num_cues,)
    # output of the decoder:
    # "content": (num_categories,)

    def __init__(
        self,
        spikes: jax.Array,
        timeline: Timeline,
        *,
        key: jax.Array,
    ):
        self.spikes = spikes
        self.num_categories = timeline.num_categories
        num_cues = timeline.num_cues
        num_timesteps, num_neurons = spikes.shape
        input_dim = num_timesteps * num_neurons + num_cues

        self.linear = eqx.nn.Linear(
            input_dim,
            self.num_categories,
            key=key,
        )

    def predict_category(self, prompt: jax.Array) -> jax.Array:
        spikes_flat = jnp.ravel(self.spikes)
        x = jnp.concatenate([spikes_flat, prompt], axis=0)

        logits = self.linear(x)
        category_idx = jnp.argmax(logits)

        return jax.nn.one_hot(category_idx, self.num_categories)
