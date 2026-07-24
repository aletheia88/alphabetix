import jax
import jax.numpy as jnp

from ..module import Module
from ..utils import sigmoid_through_threshold
from .neuron_model import Neuron, NeuronModel


class ConductanceNeuronModel(NeuronModel):
    """A conductance-based leaky-integrate-and-fire neuron.

    Models the refractory period as a change to the effective conductance.
    """

    membrane_capacitance: jnp.float32 = Module.static(default=200.0)  # pF
    leaky_reversal_potential: jnp.float32 = Module.static(default=-70.0)  # mV
    leaky_conductance: jnp.float32 = Module.static(default=10.0)  # nS
    spiking_threshold: jnp.float32 = Module.static(default=-50.0)  # mV

    refractory_potential: jnp.float32 = Module.static(default=-90.0)  # mV
    refractory_conductance_inc: jnp.float32 = Module.static(default=100.0)  # nS
    tau_refractory: jnp.float32 = Module.static(default=0.5)  # msec

    class Neuron(Neuron):
        """States of a neuron."""

        # static states
        position: jax.Array
        sign: int  # +1 for excitatory, -1 for inhibitory
        type: int  # 1: exc 2: som 3: pv 4: vip

        # dynamic states
        spike: jnp.float32 = 0.0  # either 1.0 or 0.0
        activation: jnp.float32 = 0.0
        current: jnp.float32 = 0.0
        voltage: jnp.float32 = -60.0
        refractory_conductance: jnp.float32 = 0.0

    def update(
        self,
        neuron,  # the state
        activation,
        current,
        dt,
    ):
        I = current

        V = neuron.voltage
        g_R = neuron.refractory_conductance

        E_L = self.leaky_reversal_potential
        E_R = self.refractory_potential
        C_m = self.membrane_capacitance
        g_L = self.leaky_conductance

        # update voltage
        dV = dt * (-g_L * (V - E_L) - g_R * (V - E_R) + I) / C_m
        V += dV

        # spike?
        spike = sigmoid_through_threshold(V, self.spiking_threshold)

        # update refractory conductance
        dg_R = (
            dt * (-g_R / self.tau_refractory) + spike * self.refractory_conductance_inc
        )
        g_R += dg_R

        return neuron.replace(
            spike=spike,
            activation=activation,
            current=current,
            voltage=V,
            refractory_conductance=g_R,
        )
