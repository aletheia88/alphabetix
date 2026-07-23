import jax.numpy as jnp

from ..module import Module
from ..utils import sigmoid_through_threshold


class NeuronModel(Module):
    membrane_capacitance: jnp.float32 = Module.static(default=200.0)  # pF
    leaky_reversal_potential: jnp.float32 = Module.static(default=-70.0)  # mV
    spiking_threshold: jnp.float32 = Module.static(default=-50.0)  # mV

    tau_refractory: jnp.float32 = Module.static(default=2.0)  # msec
    reset_voltage: jnp.float32 = Module.static(default=-60.0)  # mV

    def update(
        self,
        neuron,  # the state
        activation,
        current,
        utilization,
        resource,
        dt,
    ):
        c_m = self.membrane_capacitance

        is_refractory = neuron.refractory_time_remaining > 0.0

        voltage_pre_spike = (
            neuron.voltage
            - (
                (dt / neuron.tau_membrane)
                * (neuron.voltage - self.leaky_reversal_potential)
            )
            - (dt / c_m) * current
        )
        candidate_spike = sigmoid_through_threshold(
            voltage_pre_spike,
            self.spiking_threshold,
        )
        spike = candidate_spike * jnp.logical_not(is_refractory)
        # update refractory period timer
        refractory_time_remaining = jnp.where(
            spike > 0.0,
            self.tau_refractory,
            jnp.maximum(0.0, neuron.refractory_time_remaining - dt),
        )
        should_reset = jnp.logical_or(is_refractory, spike > 0.0)
        voltage = jnp.where(
            should_reset,
            self.reset_voltage,
            voltage_pre_spike,
        )

        return neuron.replace(
            spike=spike,
            activation=activation,
            current=current,
            voltage=voltage,
            refractory_time_remaining=refractory_time_remaining,
            utilization=utilization,
            resource=resource,
        )
