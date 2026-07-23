from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .models import Model, Network, Neuron
from .module import Module
from .record import Probes
from .simulate import run_simulation


class BatchLog(Module):
    """A data class to store training details for a single batch."""

    connectivity: jax.Array
    input_current: jax.Array
    loss: jax.Array
    # raw gradients
    connectivity_grads: jax.Array
    # optimizer-transformed weight updates
    connectivity_updates: jax.Array


@partial(eqx.filter_jit)
def train_step(
    model: Model,
    loss_function: Callable[[dict[str, jax.Array]], jax.Array],
    initial_network: Network,
    initial_neurons: Neuron,
    probes: Probes,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    params, static = model.partition()

    def batch_loss_grad(params):
        model = eqx.combine(params, static)
        measurements, _, _ = run_simulation(
            model,
            initial_network,
            initial_neurons,
            probes,
        )
        # TODO: split key by `batch_size` and pass to `simulation_loss`
        loss = loss_function(measurements)
        batch_log = log_iteration(model, loss)
        return loss, (batch_log, measurements)

    (loss, (batch_log, measurements)), grads = eqx.filter_value_and_grad(
        batch_loss_grad, has_aux=True
    )(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = _constrain_connectivity(params)
    model = eqx.combine(params, static)

    connectivity_grads = grads.network_model.connectivity
    connectivity_updates = updates.network_model.connectivity
    batch_log = batch_log.replace(
        connectivity_grads=connectivity_grads,
        connectivity_updates=connectivity_updates,
    )

    return model, opt_state, loss, batch_log, measurements


def log_iteration(model, loss):
    updated_connectivity = model.network_model.connectivity
    updated_inputs = model.input_model.compute_currents(model.dt)

    batch_log = BatchLog(
        connectivity=updated_connectivity,
        input_current=updated_inputs,
        loss=loss,
        connectivity_grads=jnp.zeros_like(model.network_model.connectivity),
        connectivity_updates=jnp.zeros_like(model.network_model.connectivity),
    )
    return batch_log


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
