import equinox as eqx
import jax

from ..module import Module


class SensoryModel(Module):
    layer: eqx.nn.Linear
    num_categories: int = Module.static()
    num_neurons: int = Module.static()

    def __init__(
        self,
        num_categories: int,
        num_neurons: int,
        *,
        key: jax.Array,
    ):
        self.num_categories = num_categories
        self.num_neurons = num_neurons
        self.layer = eqx.nn.Linear(num_categories, num_neurons, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim != 1:
            raise ValueError(
                f"SensoryModel expects a 1D input of shape ({self.num_categories},), "
                f"got shape {x.shape}."
            )
        if x.shape[0] != self.num_categories:
            raise ValueError(
                f"Expected input dimension {self.num_categories}, got {x.shape[0]}."
            )

        return self.layer(x)
