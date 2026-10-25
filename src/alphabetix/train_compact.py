"""Training without detailed logging or simulation outputs."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .simulate import run_simulation


def train_step(
    params,
    fixed,
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
):
    """Update selected parameters, then apply experiment-provided bounds."""
    # separate frozen arrays and python metadata
    fixed_arrays, fixed_static = eqx.partition(fixed, eqx.is_array)
    return _train_step(
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
    )


@partial(
    jax.jit,
    static_argnames=(
        "fixed",
        "probes",
        "optimizer",
        "loss_function",
    ),
)
def _train_step(
    params,
    fixed_arrays,
    fixed,
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
):
    def objective(candidate):
        model = eqx.combine(candidate, fixed_arrays, fixed)

        def trial_loss(noise_key):
            measurements, _, _ = run_simulation(
                model,
                initial_network.replace(noise_key=noise_key),
                initial_neurons,
                probes,
                timeline_inputs,
            )
            return loss_function(model, measurements)

        return jnp.mean(jax.lax.map(trial_loss, noise_keys))

    loss, gradients = jax.value_and_grad(objective)(params)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    # only constrain fields selected for training; frozen fields remain None.
    for model_name, model_constraints in constraints.items():
        model = getattr(params, model_name)
        constrained = {
            name: jnp.clip(getattr(model, name), *bounds)
            for name, bounds in model_constraints.items()
            if getattr(model, name) is not None
        }
        params = params.replace(**{model_name: model.replace(**constrained)})

    return params, optimizer_state, loss
