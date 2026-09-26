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

        key_layer, key_weight = jax.random.split(key)
        self.layer = eqx.nn.Linear(num_cues, num_neurons, key=key_layer)
        weights = jax.random.uniform(
            key_weight,
            shape=(num_neurons, num_cues),
            minval=-100.0,
            maxval=0.0,
        )
        self.layer = eqx.tree_at(
            lambda layer: layer.weight,
            self.layer,
            weights,
        )

    def __call__(self, temporal_encoding: jax.Array) -> jax.Array:
        return self.layer(temporal_encoding)
