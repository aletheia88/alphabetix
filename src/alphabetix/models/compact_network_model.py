from dataclasses import fields

import jax
import jax.numpy as jnp

from .network_model import NetworkModel


class CompactNetworkModel(NetworkModel):
    """Reuse the original neuron stepping with type-aggregated conductances."""

    @classmethod
    def from_model(cls, model):
        """Copy the original model's fields without changing parameter values."""
        return cls(
            **{field.name: getattr(model, field.name) for field in fields(NetworkModel)}
        )

    def _compute_activations_and_currents(self, neurons, network, input_currents, dt):
        groups = network.neuron_groups
        exc_mask = neurons.sign > 0
        group_exc = network.group_signs > 0
        membership = jax.nn.one_hot(
            groups, network.group_signs.size, dtype=neurons.spike.dtype
        )
        spikes = neurons.spike[:, None] * membership
        plain_increment = self.connectivity @ spikes
        stp_increment = self.connectivity @ (
            spikes * (neurons.utilization * neurons.resource)[:, None]
        )
        # short-term plasticity only applies to exc-exc connections.
        increment = jnp.where(
            exc_mask[:, None] & group_exc[None, :],
            stp_increment,
            plain_increment,
        )
        synapse_taus = network.synapse_taus[groups]
        synapse_activations = (
            1 - dt / synapse_taus
        ) * network.synapse_activations + increment

        exc_activations = jnp.sum(synapse_activations * group_exc, axis=1)
        inh_activations = jnp.sum(
            synapse_activations * (network.group_signs < 0), axis=1
        )
        activations = exc_activations + inh_activations

        fired_exc = neurons.spike * exc_mask
        u_temp = (
            neurons.utilization + dt * (self.u_total - neurons.utilization) / self.tau_f
        )
        u = u_temp + self.facilitation_increment * (1 - u_temp) * fired_exc
        x_temp = neurons.resource + dt * (1 - neurons.resource) / self.tau_d
        x = x_temp - u_temp * x_temp * fired_exc

        key0, key1 = jax.random.split(network.noise_key)
        previous_bg_currents = neurons.background_current
        noise = jax.random.normal(
            key1, shape=previous_bg_currents.shape, dtype=previous_bg_currents.dtype
        )
        decay = jnp.exp(-dt / self.tau_bg)
        noise_scale = self.sigma_bg_current * jnp.sqrt(
            -jnp.expm1(-2.0 * dt / self.tau_bg)
        )
        bg_currents = (
            self.mu_bg_current
            + decay * (previous_bg_currents - self.mu_bg_current)
            + noise_scale * noise
        )
        exc_currents = exc_activations * (neurons.voltage - self.exc_reversal_potential)
        inh_currents = inh_activations * (neurons.voltage - self.inh_reversal_potential)
        synaptic_currents = exc_currents + inh_currents
        currents = synaptic_currents + input_currents + bg_currents

        next_network = network.replace(
            synapse_activations=synapse_activations,
            noise_key=key0,
        )
        updated_neurons = neurons.replace(
            activation=activations,
            background_current=bg_currents,
            synaptic_current=synaptic_currents,
            task_input_current=input_currents,
            current=currents,
            utilization=u,
            resource=x,
        )

        return next_network, updated_neurons
