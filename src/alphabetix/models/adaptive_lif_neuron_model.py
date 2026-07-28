import jax.numpy as jnp

from ..module import Module
from ..utils import sigmoid_through_threshold
from .neuron_model import NeuronModel


class AdaptiveNeuronModel(NeuronModel):
    membrane_capacitance: jnp.float32 = Module.static(default=200.0)  # pF
    leaky_reversal_potential: jnp.float32 = Module.static(default=-70.0)  # mV
    spiking_threshold: jnp.float32 = Module.static(default=-50.0)  # mV
    reset_voltage: jnp.float32 = Module.static(default=-60.0)  # mV

    # parameters for relative refractory effect
    tau_sra: float = Module.static(default=8.0)  # msec
    tau_refractory: jnp.float32 = Module.static(default=2.0)  # msec
    delta_g_sra: jnp.float32 = Module.static(default=100.0)  # nS
    sra_reversal_potential: jnp.float32 = Module.static(default=-90.0)  # mV

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

        g_sra_decayed = neuron.g_sra * jnp.exp(-dt / self.tau_sra)
        sra_current = g_sra_decayed * (neuron.voltage - self.sra_reversal_potential)

        voltage = (
            neuron.voltage
            - (
                (dt / neuron.tau_membrane)
                * (neuron.voltage - self.leaky_reversal_potential)
            )
            - (dt / c_m) * (current + sra_current)
        )

        candidate_spike = sigmoid_through_threshold(
            voltage,
            self.spiking_threshold,
        )
        spike = candidate_spike * jnp.logical_not(is_refractory)
        g_sra = g_sra_decayed + self.delta_g_sra * spike

        # update refractory period timer
        refractory_time_remaining = jnp.where(
            spike > 0.0,
            self.tau_refractory,
            jnp.maximum(0.0, neuron.refractory_time_remaining - dt),
        )

        return neuron.replace(
            spike=spike,
            activation=activation,
            current=current,
            voltage=voltage,
            refractory_time_remaining=refractory_time_remaining,
            utilization=utilization,
            resource=resource,
            g_sra=g_sra,
        )
