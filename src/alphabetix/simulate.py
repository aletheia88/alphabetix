import jax
import jax.numpy as jnp

from .models import Model
from .record import Probes


def run_simulation(
    model: Model,
    dt: float,
    inputs: jax.Array,  # delete later
    probes: Probes | None = None,
):
    if probes is None:
        probes = Probes()  # empty

    # compute inputs
    # inputs = model.input_model.compute_activations(dt)

    # record initial measurement
    initial_measurements = probes.process(model.initial_neurons)
    neuron_model = model.neuron_model

    def scan_fn(carry, input_t):
        network, neurons = carry

        next_network, next_neurons = model.network_model.step(
            neuron_model,
            neurons,
            network,
            input_t,
            dt,
        )
        next_carry = (next_network, next_neurons)

        # record measurement
        measurement_t = probes.process(neurons)

        return next_carry, measurement_t

    init_carry = (model.initial_network, model.initial_neurons)
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
