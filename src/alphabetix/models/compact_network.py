"""Network state with conductances aggregated by presynaptic neuron type."""

import jax
import jax.numpy as jnp

from .network import Network


class CompactNetwork(Network):
    """Store synapse_taus as (K, K) and synapse_activations as (N, K).

    `neuron_groups` maps neurons to sorted type indices;
    `group_signs` identifies excitatory and inhibitory groups;
    `membership` must stay fixed during a run.
    """

    neuron_groups: jax.Array
    group_signs: jax.Array

    @classmethod
    def from_network(cls, network, neurons):
        """Individual synaptic histories are lost."""
        _, first, groups = jnp.unique(
            neurons.type,
            return_index=True,
            return_inverse=True,
        )
        group_signs = neurons.sign[first]
        type_taus = network.synapse_taus[jnp.ix_(first, first)]
        membership = jax.nn.one_hot(
            groups,
            len(first),
            dtype=network.synapse_activations.dtype,
        )

        return cls(
            synapse_taus=type_taus,
            synapse_activations=network.synapse_activations @ membership,
            noise_key=network.noise_key,
            neuron_groups=groups,
            group_signs=group_signs,
        )
