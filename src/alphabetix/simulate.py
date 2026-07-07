import jax
import jax.numpy as jnp

from .models import Model, Network, Neuron
from .record import Probes


def run_simulation(
    model: Model,
    inputs: jax.Array,
    initial_network: Network,
    initial_neurons: Neuron,
    probes: Probes | None = None,
):
    if probes is None:
        probes = Probes()  # empty

    # record initial measurement
    initial_measurements = probes.process(initial_neurons)
    neuron_model = model.neuron_model
    dt = model.dt

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
        measurement_t = probes.process(next_neurons)

        return next_carry, measurement_t

    init_carry = (initial_network, initial_neurons)
    (final_network, final_neurons), measurements = jax.lax.scan(
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
    return measurements, final_network, final_neurons
