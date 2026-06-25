# from alphabetix.probe import Probes
import jax
import jax.numpy as jnp

from .models import NetworkModel, NeuronModel
from .record import Probes


def run_simulation(
    network: NetworkModel,
    neurons: NeuronModel,
    dt: float,
    inputs: jax.Array,  # precomputed from task
    probes: Probes | None = None,
):
    if probes is None:
        probes = Probes()  # empty

    # record initial measurement
    initial_measurements = probes.process(neurons)

    def scan_fn(carry, input_t):
        network, neurons = carry

        next_network, next_neurons = network.step(
            neurons,
            input_t,
            dt,
        )
        next_carry = (next_network, next_neurons)

        # record measurement
        measurement_t = probes.process(neurons)

        return next_carry, measurement_t

    init_carry = (network, neurons)
    _, measurements = jax.lax.scan(
        scan_fn,
        init_carry,
        inputs,
    )
    measurements = jax.tree.map(
        lambda initial, history: jnp.concatenate(
            [initial[None, ...], history],
            axis=0,
        ),
        initial_measurements,
        measurements,
    )
    return measurements
