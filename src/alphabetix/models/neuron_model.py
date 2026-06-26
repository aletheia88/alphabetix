import jax.numpy as jnp

from ..utils import straight_through_threshold
from .constants import Constants


class NeuronModel:
    @classmethod
    def update(
        self,
        neuron,  # the state
        activation,
        current,
        utilization,
        resource,
        dt,
    ):
        c_m = Constants.membrane_capacitance

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
            Constants.tau_refractory,
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
            voltage=voltage,
            refractory_time_remaining=refractory_time_remaining,
            utilization=utilization,
            resource=resource,
        )
