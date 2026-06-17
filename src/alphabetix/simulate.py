import jax
import jax.numpy as jnp

from .models import NetworkModel, NeuronModel


def run_simulation(
    network: NetworkModel,
    neurons: NeuronModel,
    dt: float,
    inputs: jax.Array,  # precomputed from task
):
    def scan_fn(carry, inputs):
        network, neurons = carry

        neural_network, neurons = network.step(
            neurons,
            inputs,
            dt,
        )
        next_carry = (neural_network, neurons)
        next_state = neurons

        return next_carry, next_state

    init_carry = (network, neurons)
    _, neurons = jax.lax.scan(
        scan_fn,
        init_carry,
        inputs,
    )

    # add initial neurons to all other neurons returned from scan
    neurons = jax.tree.map(
        lambda a, i: jnp.insert(a, 0, i, axis=0), neurons, init_carry[1]
    )

    return neurons
