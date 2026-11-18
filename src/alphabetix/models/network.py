import jax

from ..module import Module


class Network(Module):
    synapse_taus: jax.Array  # (num_neurons, num_neurons)
    synapse_activations: jax.Array  # (num_neurons, num_neurons)
    noise_key: jax.Array
    # dimensionless NMDA receptor state variables, shape (2, N);
    # 0: excitatory source; 1: inhibitory source.
    nmda_rise: jax.Array | None = Module.static(default=None, kw_only=True)
    nmda_gate: jax.Array | None = Module.static(default=None, kw_only=True)
