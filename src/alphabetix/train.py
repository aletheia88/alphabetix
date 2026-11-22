from collections.abc import Callable, Sequence
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .models import Model, Network, Neuron, TimelineInputs
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


def train_step(
    params: Model,
    fixed: Model,
    initial_network: Network,
    initial_neurons: Neuron,
    probes: Probes,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    timeline_inputs: TimelineInputs,
    loss_function: Callable[
        [Model, dict[str, jax.Array]],
        jax.Array | tuple[jax.Array, StepLog],
    ],
    noise_keys: jax.Array,
    *,
    constraints: dict[str, dict[str, Sequence[float | jax.Array]]],
    log_fields: Sequence[str] = (),
):
    """Train on one timeline or a leading batch of independently reset timelines.

    For batched inputs, returned measurements and states retain the trial axis.
    """
    # separate frozen arrays and python metadata
    fixed_arrays, fixed_static = eqx.partition(fixed, eqx.is_array)

    return _update_step(
        params,
        fixed_arrays,
        fixed_static,
        initial_network,
        initial_neurons,
        probes,
        optimizer,
        optimizer_state,
        timeline_inputs,
        loss_function,
        noise_keys,
        constraints=constraints,
        log_fields=tuple(log_fields),
    )


@partial(
    jax.jit,
    static_argnames=(
        "fixed",
        "probes",
        "optimizer",
        "loss_function",
        "log_fields",
    ),
)
def _update_step(
    params,
    fixed_arrays,
    fixed_static,
    initial_network,
    initial_neurons,
    probes,
    optimizer,
    optimizer_state,
    timeline_inputs,
    loss_function,
    noise_keys,
    *,
    constraints,
    log_fields,
):
    def objective(candidate):
        model = eqx.combine(candidate, fixed_arrays, fixed_static)

        # because we have B input timelines and K noise keys for batching
        # one optimizer step includes B * K simulations
        def simulate_one(noise_key):
            """Runs the entire batch containing all timelines with one noise key."""

            def simulate_trial(values):
                """Runs on one timeline with one noise key rendering."""
                inputs, key = values
                return run_simulation(
                    model,
                    initial_network.replace(noise_key=key),
                    initial_neurons,
                    probes,
                    inputs,
                )

            if timeline_inputs.category_encodings.ndim == 3:
                # category_encodings: (B, num_timesteps, num_categories)
                # where B := the number of different input timelines (or sequences)
                num_timelines = len(timeline_inputs.category_encodings)
                keys = jax.random.split(noise_key, num_timelines)
                measurements, final_network, final_neurons = jax.lax.map(
                    simulate_trial, (timeline_inputs, keys)
                )
            # simulate directly if there is a single unbatched timeline
            else:
                measurements, final_network, final_neurons = simulate_trial(
                    (timeline_inputs, noise_key)
                )
            # loss function sees all simulated trials rendered this given noise key
            result = loss_function(model, measurements)
            loss, step_log = (
                result if isinstance(result, tuple) else (result, StepLog())
            )
            return loss, (step_log, measurements, final_network, final_neurons)

        # with B timelines (sequences) and K noise keys run B * K simulations;
        # the loss function is called K times, each time on a batch containing
        # B timelines.
        losses, diagnostics = jax.lax.map(simulate_one, noise_keys)

        return jnp.mean(losses), diagnostics

    (loss, diagnostics), gradients = jax.value_and_grad(objective, has_aux=True)(params)

    batched_logs, *batched_states = diagnostics
    step_log = jax.tree.map(lambda value: jnp.mean(value, axis=0), batched_logs)
    # axis 0: the noise-key axis
    measurements, final_network, final_neurons = jax.tree.map(
        lambda value: value[0], batched_states
    )

    updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    params = apply_constraints(params, constraints)

    model = eqx.combine(params, fixed_arrays, fixed_static)
    step_log = _log_iteration(
        step_log,
        model,
        timeline_inputs,
        loss,
        gradients,
        updates,
        log_fields,
    )
    return (
        params,
        optimizer_state,
        step_log,
        measurements,
        final_network,
        final_neurons,
    )


def apply_constraints(params, constraints):
    """Project selected fields onto their allowed bounds."""

    def field(tree):
        for component in (model_name + "." + name).split("."):
            tree = getattr(tree, component)
        return tree

    for model_name, fields in constraints.items():
        for name, bounds in fields.items():
            value = field(params)
            if value is not None:
                params = eqx.tree_at(field, params, jnp.clip(value, *bounds))

    return params


def _log_iteration(
    step_log: StepLog,
    model: Model,
    timeline_inputs: TimelineInputs,
    decoder_loss: jax.Array,
    decoder_grads: Model,
    updates: Model,
    log_fields: list[str],
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
