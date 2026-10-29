"""Experiment-facing model construction and parameter selection."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .models import (
    AdaptiveNeuronModel,
    DecoderModel,
    Model,
    Network,
    NetworkModel,
    Neuron,
    TimelineInputModel,
)
from .record import Probes


@dataclass(frozen=True)
class Settings:
    """Required model and initialization settings; values belong in each experiment."""

    dt_ms: float
    counts: tuple[int, int, int, int]
    membrane_tau_ms: tuple[float, float, float, float]
    initial_voltage_mv: tuple[float, float, float, float]
    initial_utilization: float
    initial_resource: float
    initial_g_sra_ns: float
    background_mean_pa: float | tuple[float, ...]
    background_sigma_pa: float | tuple[float, ...]
    background_tau_ms: float
    synapse_tau_ms: tuple[tuple[float, ...], ...]
    u_total: float
    tau_f_ms: float
    tau_d_ms: float
    facilitation_increment: float
    adaptation_tau_ms: float
    adaptation_increment_ns: float
    sensory_weight_range: tuple[float, float]
    topdown_weight_range: tuple[float, float]
    decoder_hidden_size: int
    position_seed: int
    input_seed: int
    decoder_seed: int
    network_noise_seed: int


def build(
    dataset,
    decode_timesteps,
    connectivity,
    settings: Settings,
    probes: Probes,
):
    """Return model and initial states, using the experiment's requested probes."""
    neuron_model = AdaptiveNeuronModel(
        tau_sra=jnp.asarray(settings.adaptation_tau_ms),
        delta_g_sra=jnp.asarray(settings.adaptation_increment_ns),
    )
    counts = settings.counts
    n = sum(counts)
    types = jnp.asarray(
        [kind for kind, count in enumerate(counts, start=1) for _ in range(count)]
    )
    signs = jnp.where(types == 1, 1, -1)
    membrane_taus = jnp.asarray(
        [
            tau
            for tau, count in zip(settings.membrane_tau_ms, counts)
            for _ in range(count)
        ]
    )
    voltages = jnp.asarray(
        [
            v
            for v, count in zip(settings.initial_voltage_mv, counts)
            for _ in range(count)
        ],
        dtype=jnp.float32,
    )
    spikes = (voltages >= neuron_model.spiking_threshold).astype(jnp.float32)
    position_key = jax.random.PRNGKey(settings.position_seed)
    input_key = jax.random.PRNGKey(settings.input_seed)
    sensory_key, topdown_key = jax.random.split(input_key)
    decoder_key = jax.random.PRNGKey(settings.decoder_seed)

    type_tau = jnp.asarray(settings.synapse_tau_ms, dtype=jnp.float32)
    synapse_taus = type_tau[types[:, None] - 1, types[None, :] - 1]
    network = Network(
        synapse_taus=synapse_taus,
        synapse_activations=jnp.zeros((n, n)),
        noise_key=jax.random.PRNGKey(settings.network_noise_seed),
    )
    network_model = NetworkModel(
        mu_bg_current=_vector(settings.background_mean_pa, n),
        sigma_bg_current=_vector(settings.background_sigma_pa, n),
        connectivity=jnp.asarray(connectivity, dtype=jnp.float32),
        u_total=jnp.asarray(settings.u_total),
        tau_f=jnp.asarray(settings.tau_f_ms),
        tau_d=jnp.asarray(settings.tau_d_ms),
        facilitation_increment=jnp.asarray(settings.facilitation_increment),
    )

    inputs = TimelineInputModel(
        vocabulary_size=dataset.vocabulary_size,
        sequence_size=dataset.sequence_size,
        num_exc_neurons=counts[0],
        num_inh_neurons=n - counts[0],
        key=input_key,
    )
    inputs = inputs.replace(
        sensory_model=inputs.sensory_model.replace(
            layer=_init_linear(
                inputs.sensory_model.layer,
                settings.sensory_weight_range,
                sensory_key,
            )
        ),
        topdown_model=inputs.topdown_model.replace(
            layer=_init_linear(
                inputs.topdown_model.layer,
                settings.topdown_weight_range,
                topdown_key,
            )
        ),
    )
    decoder = DecoderModel(
        decode_timesteps,
        n,
        dataset.vocabulary_size,
        settings.decoder_hidden_size,
        key=decoder_key,
    )
    model = Model(decoder, inputs, network_model, neuron_model, settings.dt_ms)

    initial_neurons = jax.vmap(Neuron)(
        position=jax.random.uniform(position_key, (n, 2)),
        sign=signs,
        type=types,
        tau_membrane=membrane_taus,
        voltage=voltages,
        spike=spikes,
        refractory_time_remaining=spikes * neuron_model.tau_refractory,
        utilization=jnp.full((n,), settings.initial_utilization),
        resource=jnp.full((n,), settings.initial_resource),
        g_sra=jnp.full((n,), settings.initial_g_sra_ns),
    )
    return model, network, initial_neurons, probes


def parameter_paths(model):
    """List array parameters declared trainable by the package Modules."""
    return sorted(
        _path_name(path)
        for path, value in jax.tree_util.tree_flatten_with_path(model.partition()[0])[0]
        if eqx.is_array(value)
    )


def partition_model(model, trainable_paths=None):
    """Partition a model; optionally train only selected declared parameter paths."""
    paths, treedef = jax.tree_util.tree_flatten_with_path(model)
    declared = set(parameter_paths(model))
    trainable = None if trainable_paths is None else set(trainable_paths)
    unknown = set() if trainable is None else trainable - declared
    if unknown:
        raise ValueError(f"Unknown trainable parameter path(s): {sorted(unknown)}")
    mask = treedef.unflatten(
        [
            _path_name(path) in declared
            and (trainable is None or _path_name(path) in trainable)
            for path, _ in paths
        ]
    )
    return eqx.partition(model, mask)


def _vector(value, size):
    value = jnp.asarray(value, dtype=jnp.float32)
    return jnp.broadcast_to(value, (size,)) if value.ndim == 0 else value


def _init_linear(layer, weight_range, key):
    weight = jax.random.uniform(
        key,
        layer.weight.shape,
        minval=weight_range[0],
        maxval=weight_range[1],
    )

    return eqx.tree_at(lambda x: x.weight, layer, weight)


def _path_name(path):
    parts = []
    for key in path:
        if hasattr(key, "name"):
            parts.append(key.name)
        elif hasattr(key, "idx"):
            parts[-1] += f"[{key.idx}]"
        else:
            parts.append(str(key))
    return ".".join(parts)
