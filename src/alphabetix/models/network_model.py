import jax
import jax.numpy as jnp

from ..module import Module


def nmda_step(
    rise,
    gate,
    release,
    dt,
    tau_rise,
    tau_decay,
    alpha,
):
    """Update the NMDA rise variable x and channel gate s by one timestep.

    Args:
        rise: Current rise variable x, before incoming release.
        gate: Current NMDA channel gate s.
        release: Dimensionless release amount arriving at the step's start,
            optionally scaled by short-term plasticity. Added directly to x
            without a factor of dt.
        dt: Timestep duration in ms.
        tau_rise: Decay time constant of x in ms.
        tau_decay: Decay time constant of s in ms.
        alpha: Channel opening rate coefficient in inverse ms.

    Returns:
        Tuple (rise_next, gate_next) at the end of the timestep.
    """
    rise_start = rise + release
    drive = alpha * rise_start * jnp.exp(-0.5 * dt / tau_rise)
    next_rise = rise_start * jnp.exp(-dt / tau_rise)

    rate = 1 / tau_decay + drive
    next_gate = gate * jnp.exp(-dt * rate) + drive / rate * (-jnp.expm1(-dt * rate))

    return next_rise, next_gate


class NetworkModel(Module):
    # background is fixed during task training.
    sigma_bg_current: jax.Array = Module.static()
    mu_bg_current: jax.Array = Module.static()
    tau_bg: jnp.float32 = Module.static(default=10.5, kw_only=True)

    # nS; rows postsynaptic, columns presynaptic.
    connectivity: jax.Array = Module.param()
    u_total: jax.Array = Module.param()
    tau_f: jax.Array | None = Module.param()
    tau_d: jax.Array | None = Module.param()
    facilitation_increment: jnp.float32 = Module.param()

    exc_reversal_potential: jnp.float32 = Module.static(default=0.0)
    inh_reversal_potential: jnp.float32 = Module.static(default=-70.0)

    nmda_enabled: bool = Module.static(default=True)
    exponential_decay_enabled: bool = Module.static(default=True)
    stp_enabled: bool = Module.static(default=True)

    # NMDA- and AMPA-mediated excitatory synapses
    ampa_gain: jax.Array | float = Module.static(default=1.0, kw_only=True)
    nmda_gain: jax.Array | float = Module.static(default=0.0, kw_only=True)
    tau_nmda_rise_ms: float = Module.static(default=2.0, kw_only=True)
    tau_nmda_decay_ms: float = Module.static(default=100.0, kw_only=True)
    nmda_alpha_per_ms: float = Module.static(default=0.5, kw_only=True)
    magnesium_mm: float = Module.static(default=1.0, kw_only=True)

    connectivity_mask: jax.Array | None = Module.static(default=None, kw_only=True)

    @property
    def effective_connectivity(self):
        if self.connectivity_mask is None:
            return self.connectivity

        return jnp.where(self.connectivity_mask, jnp.maximum(self.connectivity, 0), 0)

    def synapse_decay(self, dt, taus):
        return jnp.exp(-dt / taus) if self.exponential_decay_enabled else 1 - dt / taus

    def step(self, neuron_model, neurons, network, inputs, dt):
        next_network, updated = self._compute_activations_and_currents(
            neurons, network, inputs, dt
        )
        next_neurons = jax.vmap(neuron_model.update, in_axes=(0, None))(updated, dt)
        return next_network, next_neurons

    def _compute_activations_and_currents(self, neurons, network, input_currents, dt):
        exc = neurons.sign > 0
        inh = neurons.sign < 0

        plain_increment = self.effective_connectivity * neurons.spike
        increment = plain_increment

        if self.stp_enabled:
            increment = jnp.where(
                exc[:, None] & exc[None, :],
                plain_increment * (neurons.utilization * neurons.resource),
                plain_increment,
            )
        activations = (
            self.synapse_decay(dt, network.synapse_taus) * network.synapse_activations
            + increment
        )
        return self._update_activations_and_currents(
            neurons,
            network,
            input_currents,
            dt,
            activations,
            jnp.sum(activations * exc, axis=1),
            jnp.sum(activations * inh, axis=1),
        )

    def _update_activations_and_currents(
        self,
        neurons,
        network,
        input_currents,
        dt,
        synapses,
        exc_activations,
        inh_activations,
    ):
        """Shared state/current update for dense and compact backends."""
        exc = neurons.sign > 0
        fired_exc = neurons.spike * exc
        u_temp = (
            neurons.utilization + dt * (self.u_total - neurons.utilization) / self.tau_f
        )
        u = u_temp + self.facilitation_increment * (1 - u_temp) * fired_exc
        x_temp = neurons.resource + dt * (1 - neurons.resource) / self.tau_d
        x = x_temp - u_temp * x_temp * fired_exc

        nmda_g = jnp.zeros_like(exc_activations)
        nmda_rise, nmda_gate = network.nmda_rise, network.nmda_gate

        if self.nmda_enabled:
            if nmda_rise is None or nmda_gate is None:
                raise ValueError("NMDA requires initialized rise and gate states.")

            if self.stp_enabled:
                stp_increment = fired_exc * neurons.utilization * neurons.resource

            next_rise, next_gate = nmda_step(
                nmda_rise,
                nmda_gate,
                jnp.stack([stp_increment, fired_exc]),
                dt,
                self.tau_nmda_rise_ms,
                self.tau_nmda_decay_ms,
                self.nmda_alpha_per_ms,
            )
            weighted = self.effective_connectivity @ next_gate.T
            block = 1 / (
                1 + self.magnesium_mm * jnp.exp(-0.062 * neurons.voltage) / 3.57
            )
            nmda_g = (
                self.nmda_gain * jnp.where(exc, weighted[:, 0], weighted[:, 1]) * block
            )

        key0, key1 = jax.random.split(network.noise_key)
        previous_bg = neurons.background_current
        noise = jax.random.normal(
            key1, shape=previous_bg.shape, dtype=previous_bg.dtype
        )
        decay = jnp.exp(-dt / self.tau_bg)
        noise_scale = self.sigma_bg_current * jnp.sqrt(
            -jnp.expm1(-2 * dt / self.tau_bg)
        )
        bg = (
            self.mu_bg_current
            + decay * (previous_bg - self.mu_bg_current)
            + noise_scale * noise
        )
        ampa_g = self.ampa_gain * exc_activations
        ampa_current = ampa_g * (neurons.voltage - self.exc_reversal_potential)
        nmda_current = nmda_g * (neurons.voltage - self.exc_reversal_potential)
        inh_current = inh_activations * (neurons.voltage - self.inh_reversal_potential)
        synaptic_current = ampa_current + inh_current + nmda_current

        return network.replace(
            synapse_activations=synapses,
            noise_key=key0,
            nmda_rise=next_rise,
            nmda_gate=next_gate,
        ), neurons.replace(
            activation=ampa_g + inh_activations + nmda_g,
            background_current=bg,
            synaptic_current=synaptic_current,
            task_input_current=input_currents,
            current=synaptic_current + input_currents + bg,
            utilization=u,
            resource=x,
            ampa_current=ampa_current,
            nmda_current=nmda_current,
            inhibitory_current=inh_current,
        )
