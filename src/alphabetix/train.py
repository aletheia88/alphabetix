from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .models import Model, Network, Neuron
from .record import Probes
from .simulate import run_simulation


class BatchLog(eqx.Module):
    """A data class to store training details for a single batch."""

    voltages: jax.Array
    network_state: Network
    neuron_state: Neuron


def simulation_step(
    model: Model,
    initial_network: Network,
    initial_neurons: Neuron,
    probes: Probes,
):
    # compute inputs
    # inputs = model.input_model.compute_currents(model.dt)  # TODO: add key to inputs
    inputs = jnp.zeros((30, 3))

    return run_simulation(model, inputs, initial_network, initial_neurons, probes)


@partial(eqx.filter_jit)
def train_step(
    model: Model,
    initial_network: Network,
    initial_neurons: Neuron,
    probes: Probes,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    params, static = model.partition()

    def batch_loss_grad(params):
        model = eqx.combine(params, static)
        measurements, final_network, final_neurons = simulation_step(
            model,
            initial_network,
            initial_neurons,
            probes,
        )
        # TODO: split key by `batch_size` and pass to `simulation_loss`
        loss, batch_log = simulation_loss(measurements, final_network, final_neurons)
        return loss, batch_log

    (loss, batch_log), grads = eqx.filter_value_and_grad(batch_loss_grad, has_aux=True)(
        params
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = _constrain_connectivity(params)
    model = eqx.combine(params, static)

    return model, opt_state, loss, batch_log


def simulation_loss(measurements, final_network, final_neurons):
    voltages = measurements["voltage"]
    exc_voltages = voltages[:5, :2]
    inh_voltages = voltages[:5, -1]
    target_voltage = -50.0
    loss = jnp.mean((inh_voltages - target_voltage) ** 2)
    batch_log = BatchLog(
        voltages=voltages,
        network_state=final_network,
        neuron_state=final_neurons,
    )
    return loss, batch_log


def _constrain_connectivity(
    params,
    connection_mask=None,
    min_value=0.0,
):
    connectivity = params.network_model.connectivity
    connectivity = jnp.maximum(connectivity, min_value)

    if connection_mask is not None:
        connectivity = connectivity * connection_mask

    return eqx.tree_at(
        lambda m: m.network_model.connectivity,
        params,
        connectivity,
    )
