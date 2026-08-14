import equinox as eqx
import jax
import jax.numpy as jnp

from ..module import Module


class DecoderModel(Module):
    layers: tuple
    timesteps: jax.Array = Module.static()
    num_timesteps: int = Module.static()
    num_neurons: int = Module.static()
    num_categories: int = Module.static()
    hidden_size: int = Module.static()

    def __init__(
        self,
        timesteps: jax.Array,
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
        self.hidden_size = hidden_size

        input_size = self.num_timesteps * num_neurons
        k1, k2 = jax.random.split(key)

        self.layers = (
            eqx.nn.Linear(input_size, hidden_size, key=k1),
            eqx.nn.Linear(hidden_size, num_categories, key=k2),
        )

    def __call__(self, spikes: jax.Array) -> jax.Array:
        x = jnp.ravel(spikes)
        x = self.layers[0](x)
        x = jax.nn.relu(x)
        logits = self.layers[1](x)

        return logits

    def predict_category(self, spikes: jax.Array) -> jax.Array:
        logits = self(spikes)
        category_idx = jnp.argmax(logits)

        return jax.nn.one_hot(category_idx, self.num_categories)
