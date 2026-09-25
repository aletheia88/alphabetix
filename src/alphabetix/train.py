from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .models import DecoderModel, Model, Network, Neuron, TimelineInputs
from .module import Module
from .record import Probes
from .simulate import run_simulation


class StepLog(Module):
    """A data class to store training details for a single batch."""

    connectivity: jax.Array | None = None
    task_input_current: jax.Array | None = None

    # training objectives
    decoder_loss: jax.Array | None = None
    decoder_probs: jax.Array | None = None
    decoder_targets: jax.Array | None = None

    # diagnostics for homeostasis
    mu_bg_current: jax.Array | None = None
    sigma_bg_current: jax.Array | None = None

    # raw gradients
    connectivity_grads: jax.Array | None = None
    mu_bg_current_grads: jax.Array | None = None
    sigma_bg_current_grads: jax.Array | None = None
    sensory_model_grads: jax.Array | None = None
    topdown_model_grads: jax.Array | None = None
    decoder_model_grads: jax.Array | None = None

    # optimizer-transformed updates
    connectivity_updates: jax.Array | None = None
    mu_bg_current_updates: jax.Array | None = None
    sigma_bg_current_updates: jax.Array | None = None
    sensory_model_updates: jax.Array | None = None
    topdown_model_updates: jax.Array | None = None
    decoder_model_updates: jax.Array | None = None


@partial(
    jax.jit,
    static_argnames=(
        "static",
        "decoder_loss_function",
        "probes",
        "optimizer",
        "log_fields",
    ),
)
def train_step(
    params: Model,
    static: Model,
    decoder_loss_function: Callable[
        [DecoderModel, jax.Array, jax.Array], tuple[jax.Array, StepLog]
    ],
    initial_network: Network,
    initial_neurons: Neuron,
    probes: Probes,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    timeline_inputs: TimelineInputs,
    target: jax.Array,
    log_fields: tuple[str, ...],
    noise_keys: jax.Array,
):
    def decoder_loss_grad(params):
        model = eqx.combine(params, static)

        def simulate_one(noise_key):
            """Simulate full-trial dynamics for once.

            Each simulation receives everything the same; i.e., timeline
            inputs, model parameters, network and neurons states.
            But a different OU-noise realization.
            """
            simulation_initial_network = initial_network.replace(noise_key=noise_key)

            measurements, final_network, final_neurons = run_simulation(
                model,
                simulation_initial_network,
                initial_neurons,
                probes,
                timeline_inputs,
            )
            decoder_loss, step_log = decoder_loss_function(
                model.decoder_model,
                measurements,
                target,
            )

            return decoder_loss, (step_log, measurements, final_network, final_neurons)

        (
            simulation_losses,
            (
                batched_step_logs,
                batched_measurements,
                batched_final_networks,
                batched_final_neurons,
            ),
        ) = jax.vmap(simulate_one)(noise_keys)

        # average the loss over samples within a batch
        decoder_loss = jnp.mean(simulation_losses)

        return decoder_loss, (
            batched_step_logs,
            batched_measurements,
            batched_final_networks,
            batched_final_neurons,
        )

    (
        (
            decoder_loss,
            (
                batched_step_logs,
                batched_measurements,
                batched_final_networks,
                batched_final_neurons,
            ),
        ),
        decoder_grads,
    ) = jax.value_and_grad(decoder_loss_grad, has_aux=True)(params)

    # average per-simulation decoder diagonistics
    # fields that are None remain None
    step_log = jax.tree_util.tree_map(
        lambda value: jnp.mean(value, axis=0),
        batched_step_logs,
    )

    # keep one simulation for measurements and states
    measurements = jax.tree_util.tree_map(
        lambda value: value[0],
        batched_measurements,
    )
    final_network = jax.tree_util.tree_map(
        lambda value: value[0],
        batched_final_networks,
    )
    final_neurons = jax.tree_util.tree_map(
        lambda value: value[0],
        batched_final_neurons,
    )

    updates, opt_state = optimizer.update(
        decoder_grads,
        opt_state,
        params,
    )
    params = optax.apply_updates(params, updates)

    # apply parameters constraints
    params = _constrain_connectivity(params)
    params = _constrain_bg_parameters(params)

    # log specified training outcomes / diagnostics
    model = eqx.combine(params, static)
    step_log = log_iteration(
        step_log,
        model,
        timeline_inputs,
        decoder_loss,
        decoder_grads,
        updates,
        log_fields,
    )

    return (
        params,
        opt_state,
        step_log,
        measurements,
        final_network,
        final_neurons,
    )


def log_iteration(
    step_log: StepLog,
    model: Model,
    timeline_inputs: TimelineInputs,
    decoder_loss: jax.Array,
    decoder_grads: Model,
    updates: Model,
    log_fields: tuple[str, ...],
) -> StepLog:
    """Create a StepLog containing only the requested fields."""
    value_getters = {
        # post-update network-level parameters
        "connectivity": lambda: model.network_model.connectivity,
        # objectives
        "decoder_loss": lambda: decoder_loss,
        "mu_bg_current": lambda: model.network_model.mu_bg_current,
        "sigma_bg_current": lambda: model.network_model.sigma_bg_current,
        # raw gradients
        "connectivity_grads": lambda: decoder_grads.network_model.connectivity,
        "sensory_model_grads": lambda: decoder_grads.input_model.sensory_model,
        "topdown_model_grads": lambda: decoder_grads.input_model.topdown_model,
        "decoder_model_grads": lambda: decoder_grads.decoder_model,
        "mu_bg_current_grads": lambda: (decoder_grads.network_model.mu_bg_current),
        "sigma_bg_current_grads": lambda: decoder_grads.network_model.sigma_bg_current,
        # optimizer-transformed updates
        "connectivity_updates": lambda: updates.network_model.connectivity,
        "sensory_model_updates": lambda: updates.input_model.sensory_model,
        "topdown_model_updates": lambda: updates.input_model.topdown_model,
        "decoder_model_updates": lambda: updates.decoder_model,
        "mu_bg_current_updates": lambda: updates.network_model.mu_bg_current,
        "sigma_bg_current_updates": lambda: updates.network_model.sigma_bg_current,
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

    return step_log.replace(**logged_values)


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
    sigma_bg_current = jnp.maximum(params.network_model.sigma_bg_current, min_sigma)

    return eqx.tree_at(
        lambda m: m.network_model.sigma_bg_current,
        params,
        sigma_bg_current,
    )
