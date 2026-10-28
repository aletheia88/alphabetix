import equinox as eqx
import jax
import jax.numpy as jnp

from ..module import Module


class DecoderModel(Module):
    layer: eqx.nn.Linear = Module.param()
    timesteps: tuple[int, ...] = Module.static()
    num_timesteps: int = Module.static()
    num_neurons: int = Module.static()
    num_categories: int = Module.static()

    def __init__(
        self,
        timesteps: tuple[int, ...],
        num_neurons: int,
        num_categories: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        self.timesteps = timesteps
        self.num_timesteps = len(timesteps)
        self.num_neurons = num_neurons
        self.num_categories = num_categories

        input_size = self.num_timesteps * num_neurons
        self.layer = eqx.nn.Linear(
            in_features=input_size,
            out_features=num_categories,
            key=key,
        )

    def __call__(self, spikes: jax.Array) -> jax.Array:
        x = jnp.ravel(spikes)
        logits = self.layer(x)

        return logits

    def predict_category(self, spikes: jax.Array) -> jax.Array:
        logits = self(spikes)
        category_idx = jnp.argmax(logits)

        return jax.nn.one_hot(category_idx, self.num_categories)
