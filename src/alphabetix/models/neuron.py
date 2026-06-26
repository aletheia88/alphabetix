import jax
import jax.numpy as jnp

from ..module import Module


class Neuron(Module):
    """States of a neuron."""

    # static states
    position: jax.Array
    sign: int  # +1 for excitatory, -1 for inhibitory
    type: int  # 1: exc 2: som 3: pv 4: vip
    tau_membrane: float  # membrane decay time constant, ms

    # dynamic states
    spike: jnp.float32 = 0.0  # either 1.0 or 0.0
    activation: jnp.float32 = 0.0
    current: jnp.float32 = 0.0
    voltage: jnp.float32 = -60.0

    refractory_time_remaining: jnp.float32 = 0.0  # msec

    utilization: jnp.float32 = 0.3  # u
    resource: jnp.float32 = 1.0  # x
