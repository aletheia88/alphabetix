from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .models import Model, Network, Neuron, TimelineInputs
from .module import Module
from .record import Probes
from .simulate import run_simulation, run_simulation_on_inputs


class StepLog(Module):
    """A data class to store training details for a single batch."""

    connectivity: jax.Array | None = None
    input_current: jax.Array | None = None

    # training objectives
    decoder_loss: jax.Array | None = None
    homeostasis_loss: jax.Array | None = None

    # diagnostics for homeostatic objective
    near_spiking_fraction: jax.Array | None = None
    spontaneous_firing_rate: jax.Array | None = None
    mean_bg_current: jax.Array | None = None
    sigma_bg: jax.Array | None = None

    # raw gradients
    connectivity_grads: jax.Array | None = None
    mean_bg_current_grads: jax.Array | None = None
    sigma_bg_current_grads: jax.Array | None = None

    # optimizer-transformed updates
    connectivity_updates: jax.Array | None = None
    mean_bg_updates: jax.Array | None = None
    sigma_bg_updates: jax.Array | None = None


@partial(
    jax.jit,
    static_argnames=(
        "decoder_loss_function",
        "homeostasis_loss_function",
        "probes",
        "optimizer",
        "log_fields",
    ),
)
def train_step(
    params: Model,
    static: Model,
    decoder_loss_function: Callable[[Model, jax.Array, jax.Array], jax.Array],
    homeostasis_loss_function: Callable[[Model, jax.Array], jax.Array],
    initial_network: Network,
    initial_neurons: Neuron,
    probes: Probes,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    timeline_inputs: TimelineInputs,
    target: jax.Array,
    log_fields: tuple[str, ...],
    noise_key: jax.Array,
):
    # update key for sampling noise
    initial_network = initial_network.replace(noise_key=noise_key)
    model = eqx.combine(params, static)

    def decoder_loss_grad(params):
        measurements, _, _ = run_simulation(
            model,
            initial_network,
            initial_neurons,
            probes,
            timeline_inputs,
        )
        query_timesteps = model.decoder_model.timesteps
        spikes = measurements["spike"][query_timesteps, :]
        decoder_loss = decoder_loss_function(model, spikes, target)

        return decoder_loss, measurements

    def homeostasis_loss_grad(params):
        task_inputs = model.input_model.compute_currents(timeline_inputs)
        task_inputs = jnp.zeros_like(task_inputs)

        measurements, _, _ = run_simulation_on_inputs(
            model,
            task_inputs,
            initial_network,
            initial_neurons,
            probes,
        )
        homeostasis_loss, (near_spiking_fraction, spontaneous_firing_rate) = (
            homeostasis_loss_function(model, measurements)
        )

        return homeostasis_loss, (near_spiking_fraction, spontaneous_firing_rate)

    (decoder_loss, measurements), decoder_grads = jax.value_and_grad(
        decoder_loss_grad, has_aux=True
    )(params)
    (
        (homeostasis_loss, (near_spiking_fraction, spontaneous_firing_rate)),
        background_grads,
    ) = jax.value_and_grad(homeostasis_loss_grad, has_aux=True)(params)

    # compute and apply gradient updates
    grads = _route_gradients(decoder_grads, background_grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # apply parameters constraints
    params = _constrain_connectivity(params)
    params = _constrain_bg_parameters(params)

    # log specified training outcomes / diagnostics
    step_log = log_iteration(
        model,
        timeline_inputs,
        decoder_loss,
        homeostasis_loss,
        near_spiking_fraction,
        spontaneous_firing_rate,
        decoder_grads,
        background_grads,
        updates,
        log_fields,
    )

    return params, opt_state, step_log, measurements


def _route_gradients(decoder_grads, background_grads):
    """Use background loss for computing background current parameters."""
    return eqx.tree_at(
        lambda m: (m.network_model.mean_bg_current, m.network_model.sigma_bg),
        decoder_grads,
        (
            background_grads.network_model.mean_bg_current,
            background_grads.network_model.sigma_bg,
        ),
    )


def log_iteration(
    model: Model,
    timeline_inputs: TimelineInputs,
    decoder_loss: jax.Array,
    homeostasis_loss: jax.Array,
    near_spiking_fraction: jax.Array,
    spontaneous_firing_rate: jax.Array,
    decoder_grads: Model,
    background_grads: Model,
    updates: Model,
    log_fields: tuple[str, ...],
) -> StepLog:
    """Create a StepLog containing only the requested fields."""
    value_getters = {
        # post-update network-level parameters
        "connectivity": lambda: model.network_model.connectivity,
        "input_current": lambda: model.input_model.compute_currents(timeline_inputs),
        # objectives
        "decoder_loss": lambda: decoder_loss,
        "homeostasis_loss": lambda: homeostasis_loss,
        # homeostatic diagnostics
        "near_spiking_fraction": lambda: near_spiking_fraction,
        "spontaneous_firing_rate": lambda: spontaneous_firing_rate,
        "mean_bg_current": lambda: model.network_model.mean_bg_current,
        "sigma_bg": lambda: model.network_model.sigma_bg,
        # raw gradients
        "connectivity_grads": lambda: decoder_grads.network_model.connectivity,
        "mean_bg_current_grads": lambda: (
            background_grads.network_model.mean_bg_current
        ),
        "sigma_bg_current_grads": lambda: background_grads.network_model.sigma_bg,
        # optimizer-transformed updates
        "connectivity_updates": lambda: updates.network_model.connectivity,
        "mean_bg_updates": lambda: updates.network_model.mean_bg_current,
        "sigma_bg_updates": lambda: updates.network_model.sigma_bg,
    }

    unknown_fields = set(log_fields) - set(value_getters)
    if unknown_fields:
        valid_fields = ", ".join(value_getters)
        unknown_fields = ", ".join(sorted(unknown_fields))
        raise ValueError(
            f"Unknown StepLog field(s): {unknown_fields}. "
            f"Valid fields are: {valid_fields}."
        )

    logged_values = {name: value_getters[name]() for name in log_fields}
    return StepLog(**logged_values)


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


def _constrain_bg_parameters(params, min_sigma=1e-6):
    """Keep the OU stationary standard deviation non-negative."""
    sigma_bg = jnp.maximum(params.network_model.sigma_bg, min_sigma)

    return eqx.tree_at(
        lambda m: m.network_model.sigma_bg,
        params,
        sigma_bg,
    )
