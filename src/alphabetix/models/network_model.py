import equinox as eqx
import jax
import jax.numpy as jnp

from ..module import Module
from .neuron_model import NeuronModel


class NetworkModel(Module):
    neurons: NeuronModel

    num_exc: int = eqx.field(static=True)
    num_pv: int = eqx.field(static=True)
    num_som: int = eqx.field(static=True)

    # rows: post-synaptic neurons
    # columns: pre-synaptic neurons
    connectivity: jax.Array  # (num_neurons, num_neurons)

    # LIF model constants
    tau_m: jax.Array  # dependent on cell types

    def __init__(
        self,
        neurons: NeuronModel,
        num_exc: int,
        num_pv: int,
        num_som: int,
    ):
        self.neurons = neurons
        self.num_exc = num_exc
        self.num_pv = num_pv
        self.num_som = num_som
        tau_exc = 2 / 1000
        tau_pv = 3.1 / 1000
        tau_som = 11.8 / 1000
        # self.tau_m: (num_neurons,)
        # neuron ordering: Exc -> PV -> SOM
        self.tau_m = jnp.concatenate(
            [
                jnp.full((self.num_exc,), tau_exc),
                jnp.full((self.num_pv,), tau_pv),
                jnp.full((self.num_som,), tau_som),
            ]
        )

    @property
    def num_neurons(self):
        return self.num_exc + self.num_pv + self.num_som

    def step(self, network_activations, input_activations, dt):
        # at t = 0, `network_activations` = `input_activations`
        activations = self.compute_activations(
            network_activations, input_activations, dt
        )
        next_neurons = jax.vmap(NeuronModel.update, in_axes=(0, 0, None))(
            self.neurons, activations, dt
        )
        return self.replace(neurons=next_neurons)

    def compute_activations(self, network_activations, input_activations, dt):
        network_activations = (
            1 - dt / self.tau_m
        ) * network_activations + self.connectivity @ (
            self.neurons.spike * self.neurons.sign
        )

        return network_activations + input_activations
