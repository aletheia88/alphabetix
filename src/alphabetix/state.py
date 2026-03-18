import jax

from .module import Module


class NeuronStates(Module):
    """The state of all neurons at a specific point in time."""

    positions: jax.Array  # ([t,] n, 2)
    currents: jax.Array  # ([t,] n, 4)
    voltages: jax.Array  # ([t,] n, 4)
    types: jax.Array  # ([t,] n), 0 = Exc, 1 = SOM, 2 = PV, 3 = VIP

    # connectivity -> NeuronModel
    # activations -> no need to keep around
    # parameters -> NeuronModel
