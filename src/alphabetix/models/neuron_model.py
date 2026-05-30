import jax
import jax.numpy as jnp

from ..module import Module
from ..utils import straight_through_threshold


class Constants:
    spiking_threshold: jnp.float32 = -50.0  # mV
    reset_voltage: jnp.float32 = -60.0  # mV
    exc_reversal_potential: jnp.float32 = 0.0  # mV
    inh_reversal_potential: jnp.float32 = -70.0  # mV
    leaky_reversal_potential: jnp.float32 = -70.0  # mV
    membrane_capacitance: jnp.float32 = 200.0  # pF


class NeuronModel(Module):
    position: jax.Array
    sign: int  # +1 for excitatory, -1 for inhibitory
    tau_membrane: float  # membrane decay time constant, ms
    # tau_synapse: float  # synaptic decay time constant, ms
    reversal_potential: float

    # state of neuron
    spike: jnp.float32 = 0.0  # either 1.0 or 0.0
    activation: jnp.float32 = 0.0
    current: jnp.float32 = 0.0
    voltage: jnp.float32 = -60.0

    @classmethod
    def update(
        self,
        neuron,
        activation,
        current,
        dt,
    ):
        # current = exc_current + inh_current
        c_m = Constants.membrane_capacitance

        voltage_pre_spike = (
            neuron.voltage
            - (dt / neuron.tau_membrane)
            * (neuron.voltage - Constants.leaky_reversal_potential)
            - (dt / c_m) * current
        )
        spike = straight_through_threshold(
            neuron.voltage,
            Constants.spiking_threshold,
        )
        voltage = (1.0 - spike) * (voltage_pre_spike) + spike * Constants.reset_voltage

        return neuron.replace(
            spike=spike,
            activation=activation,
            current=current,
            voltage=voltage,
        )
