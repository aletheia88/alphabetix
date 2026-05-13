import equinox as eqx
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
    tau_synapse: float  # synaptic decay time constant, ms

    # state of neuron
    spike: jnp.float32 = 0.0  # either 1.0 or 0.0
    activation: jnp.float32 = 0.0
    current: jnp.float32 = 0.0
    voltage: jnp.float32 = -70.0

    @classmethod
    def update(self, neuron, activation, current, dt):
        # TODO: reversal potential depends on presynaptic neuron type not postsynaptic
        # reversal_potential = (
        #     neuron.sign * Constants.exc_reversal_potential
        #     + (1.0 - neuron.sign) * Constants.inh_reversal_potential / 2
        # )
        # TODO: reversal potential is based on exc/inh activation source (presynaptic)
        # not based on neuron.sign (postsynaptic neuron type)

        c_m = Constants.membrane_capacitance

        spike = straight_through_threshold(neuron.voltage, Constants.spiking_threshold)
        voltage = (1.0 - spike) * (
            neuron.voltage
            - (dt / neuron.tau_membrane)
            * (neuron.voltage - Constants.leaky_reversal_potential)
            - (dt / c_m) * current
        ) + spike * Constants.reset_voltage

        return neuron.replace(
            spike=spike,
            activation=activation,
            current=current,
            voltage=voltage,
        )
