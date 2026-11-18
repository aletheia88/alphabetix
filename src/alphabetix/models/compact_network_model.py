from dataclasses import fields

import jax
import jax.numpy as jnp

from .network_model import NetworkModel


class CompactNetworkModel(NetworkModel):
    """Aggregate linear synapses by source type; keep nonlinear NMDA gates per source."""

    @classmethod
    def from_model(cls, model):
        return cls(
            **{field.name: getattr(model, field.name) for field in fields(NetworkModel)}
        )

    def _compute_activations_and_currents(self, neurons, network, input_currents, dt):
        groups = network.neuron_groups
        exc = neurons.sign > 0
        group_exc = network.group_signs > 0
        group_inh = network.group_signs < 0

        membership = jax.nn.one_hot(
            groups, network.group_signs.size, dtype=neurons.spike.dtype
        )
        spikes = neurons.spike[:, None] * membership
        plain_increment = self.effective_connectivity @ spikes
        increment = plain_increment

        if self.stp_enabled:
            stp_increment = self.effective_connectivity @ (
                spikes * (neurons.utilization * neurons.resource)[:, None]
            )
            increment = jnp.where(
                exc[:, None] & group_exc[None, :], stp_increment, plain_increment
            )

        synapses = (
            self.synapse_decay(dt, network.synapse_taus[groups])
            * network.synapse_activations
            + increment
        )
        return self._update_activations_and_currents(
            neurons,
            network,
            input_currents,
            dt,
            synapses,
            jnp.sum(synapses * group_exc, axis=1),
            jnp.sum(synapses * group_inh, axis=1),
        )
