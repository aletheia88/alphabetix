import equinox as eqx
import jax
import jax.numpy as jnp

from ..module import Module


class DecoderModel(Module):
    """Predict one category from a flattened (time bins, neurons) matrix."""

    layer: eqx.nn.Linear = Module.param()
    num_timesteps: int = Module.static()
    num_neurons: int = Module.static()
    num_categories: int = Module.static()

    def __init__(
        self,
        num_timesteps: int,
        num_neurons: int,
        num_categories: int,
        *,
        key: jax.Array,
    ):
        self.num_timesteps = num_timesteps
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


def sliding_window_rates(
    spikes,
    window_steps: int,
    stride_steps: int,
    dt: float,
):
    """Return (windows, neurons) rates in Hz; omit incomplete trailing windows.

    Window and stride are fixed integer step counts when used under JIT.
    """
    if spikes.ndim != 2 or not 1 <= window_steps <= spikes.shape[0]:
        raise ValueError(
            "Expected (time, neurons) spikes and a window within the recording."
        )
    if stride_steps < 1:
        raise ValueError("stride_steps must be positive.")

    cumulative = jnp.concatenate(
        [jnp.zeros_like(spikes[:1]), jnp.cumsum(spikes, axis=0)], axis=0
    )
    starts = jnp.arange(0, spikes.shape[0] - window_steps + 1, stride_steps)
    counts = cumulative[starts + window_steps] - cumulative[starts]

    return counts * (1000 / (window_steps * dt))
