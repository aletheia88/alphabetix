import jax
import jax.numpy as jnp

from ..module import Module
from ..utils import straight_through_threshold


class Constants:
    # leaky-integrate-fire model parameters
    spiking_threshold: jnp.float32 = -50.0  # mV
    reset_voltage: jnp.float32 = -60.0  # mV
    exc_reversal_potential: jnp.float32 = 0.0  # mV
    inh_reversal_potential: jnp.float32 = -70.0  # mV
    leaky_reversal_potential: jnp.float32 = -70.0  # mV
    membrane_capacitance: jnp.float32 = 200.0  # pF

    # short-term plasticity parameters (Mongillo et al., 2008)
    u_total: jnp.float32 = 0.3
    x_max: jnp.float32 = 1.0
    tau_f: jnp.float32 = 1600  # msec
    tau_d: jnp.float32 = 50  # msec


class NeuronModel(Module):
    position: jax.Array
    sign: int  # +1 for excitatory, -1 for inhibitory
    tau_membrane: float  # membrane decay time constant, ms
    # tau_synapse: float  # synaptic decay time constant, ms
    reversal_potential: float
    tau_refractory: float  # msec

    # state of neuron
    spike: jnp.float32 = 0.0  # either 1.0 or 0.0
    activation: jnp.float32 = 0.0
    current: jnp.float32 = 0.0
    internal_exc_current: jnp.float32 = 0.0  # remove
    input_exc_current: jnp.float32 = 0.0  # remove
    exc_current: jnp.float32 = 0.0
    inh_current: jnp.float32 = 0.0
    voltage: jnp.float32 = -60.0

    refractory_time_remaining: jnp.float32 = 0.0  # msec

    utilization: jnp.float32 = Constants.u_total  # u
    resource: jnp.float32 = Constants.x_max  # x

    @classmethod
    def update(
        self,
        neuron,
        activation,
        internal_exc_current,
        input_exc_current,
        exc_current,
        inh_current,
        utilization,
        resource,
        dt,
    ):
        c_m = Constants.membrane_capacitance

        current = exc_current + inh_current

        is_refractory = neuron.refractory_time_remaining > 0.0

        voltage_pre_spike = (
            neuron.voltage
            - (dt / neuron.tau_membrane)
            * (neuron.voltage - Constants.leaky_reversal_potential)
            - (dt / c_m) * current
        )
        candidate_spike = straight_through_threshold(
            neuron.voltage,
            Constants.spiking_threshold,
        )
        spike = candidate_spike * (1.0 - is_refractory.astype(jnp.float32))
        # update refractory period timer
        refractory_time_remaining = jnp.where(
            spike > 0.0,
            neuron.tau_refractory,
            jnp.maximum(0.0, neuron.refractory_time_remaining - dt),
        )
        should_reset = jnp.logical_or(is_refractory, spike > 0.0)
        voltage = jnp.where(
            should_reset,
            Constants.reset_voltage,
            voltage_pre_spike,
        )

        return neuron.replace(
            spike=spike,
            activation=activation,
            current=current,
            internal_exc_current=internal_exc_current,
            input_exc_current=input_exc_current,
            exc_current=exc_current,
            inh_current=inh_current,
            voltage=voltage,
            refractory_time_remaining=refractory_time_remaining,
            utilization=utilization,
            resource=resource,
        )
