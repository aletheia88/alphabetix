import equinox as eqx
import jax

from ..module import Module


class TopDownModel(Module):
    layer: eqx.nn.Linear
    num_cues: int = Module.static()
    num_neurons: int = Module.static()

    def __init__(
        self,
        num_cues: int,
        num_neurons: int,
        *,
        key: jax.Array,
    ):
        self.num_cues = num_cues
        self.num_neurons = num_neurons
        self.layer = eqx.nn.Linear(num_cues, num_neurons, key=key)

    def __call__(self, temporal_encoding: jax.Array) -> jax.Array:
        return self.layer(temporal_encoding)
