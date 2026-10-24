import equinox as eqx
import jax

from ..module import Module


class TopDownModel(Module):
    layer: eqx.nn.Linear
    topdown_token_size: int = Module.static()
    num_neurons: int = Module.static()

    def __init__(
        self,
        sequence_size: int,
        num_neurons: int,
        *,
        key: jax.Array,
    ):
        self.topdown_token_size = sequence_size
        self.num_neurons = num_neurons
        self.layer = eqx.nn.Linear(
            self.topdown_token_size,
            self.num_neurons,
            use_bias=False,
            key=key,
        )

    def __call__(self, temporal_encoding: jax.Array) -> jax.Array:
        return self.layer(temporal_encoding)
