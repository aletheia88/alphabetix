import jax
import jax.numpy as jnp

from ..module import Module
from ..utils import sigmoid_through_threshold
from .neuron_model import Neuron, NeuronModel


class LIFNeuronModel(NeuronModel):
    """Implements a simple leaky-integrate-and-fire neuron.

    Models the refractory period as a period of constant voltage.
    """

    membrane_capacitance: jnp.float32 = Module.static(default=200.0)  # pF
    leaky_reversal_potential: jnp.float32 = Module.static(default=-70.0)  # mV
    spiking_threshold: jnp.float32 = Module.static(default=-50.0)  # mV

    tau_refractory: jnp.float32 = Module.static(default=2.0)  # msec
    reset_voltage: jnp.float32 = Module.static(default=-60.0)  # mV

    class Neuron(Neuron):
        tau_membrane: float  # membrane decay time constant, ms
        refractory_conductance: jnp.float32 = 0.0
        refractory_time_remaining: jnp.float32 = 0.0  # msec
        utilization: jnp.float32 = 0.3  # u
        resource: jnp.float32 = 1.0  # x

    def update(
        self,
        neuron,  # the state
        activation,
        current,
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
            + (dt / c_m) * current
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
        )
