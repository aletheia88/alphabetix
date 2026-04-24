import equinox as eqx
import jax
import jax.numpy as jnp

from ..module import Module
from ..utils import straight_through_threshold


class Constants:
    spiking_threshold: jnp.float32 = -50  # mV
    reset_voltage: jnp.float32 = -60  # mV
    exc_reversal_potential: jnp.float32 = 0.0  # mV
    inh_reversal_potential: jnp.float32 = -70.0  # mV
    leaky_reversal_potential: jnp.float32 = -70.0  # mV
    membrane_capacitance: jnp.float32 = 0.2  # nF


class NeuronModel(Module):
    position: jax.Array = eqx.field(static=True)  # (2,)
    tau: float = eqx.field(static=True)
    sign: int = eqx.field(static=True)  # +1 for excitatory, -1 for inhibitory

    # state of neuron
    spike: jnp.float32 = 0.0  # either 1.0 or 0.0
    current: jnp.float32 = 0.0
    voltage: jnp.float32 = 0.0

    def __init__(
        self,
        position: jax.Array,
        tau: float,
        sign: int,
        spike: float = 0.0,
        current: float = 0.0,
        voltage: float = -70.0,
    ):
        self.tau = tau
        self.sign = sign

        self.spike = jnp.asarray(spike)
        self.current = jnp.asarray(current)
        self.voltage = jnp.asarray(voltage)

    @classmethod
    def update(self, neuron, activation, dt):
        # use parameters of neuron to find out what happens to current ->
        # voltage -> spike
        spike = straight_through_threshold(neuron.voltage, Constants.spiking_threshold)
        reversal_potential = (
            neuron.sign * Constants.exc_reversal_potential
            + (1.0 - neuron.sign) * Constants.inh_reversal_potential
        )
        current = activation * (neuron.voltage - reversal_potential)
        c_m = Constants.membrane_capacitance
        voltage = (1.0 - spike) * (
            neuron.voltage
            - (dt / neuron.tau) * (neuron.voltage - Constants.leaky_reversal_potential)
            + (dt / c_m) * neuron.current
        ) + spike * Constants.reset_voltage

        return neuron.replace(spike=spike, current=current, voltage=voltage)


if __name__ == "__main__":
    pv_neuron = NeuronModel(
        position=jnp.array([0, 0]), tau=3.1, sign=-1, current=10, voltage=10
    )
    som_neuron = NeuronModel(position=jnp.array([1, 0]), tau=11.8, sign=-1)
    exc_neuron = NeuronModel(position=jnp.array([0, 1]), tau=10.9, sign=1)
    activation = 1
    dt = 1
    pv_neuron = pv_neuron.update(pv_neuron, activation, dt)
    print(pv_neuron.current)
