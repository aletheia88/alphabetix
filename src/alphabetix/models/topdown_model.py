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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # number of timeslots plus 1 ("zero" / delay period)
    num_cues = 3
    num_neurons = 20
    model = TopDownModel(
        num_cues,
        num_neurons,
        key=key,
    )
    temporal_encoding = jax.nn.one_hot(1, num_cues)
    print(temporal_encoding.shape)
    y = model(temporal_encoding)
    print(y.shape)
