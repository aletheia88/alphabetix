import jax
import jax.numpy as jnp

from ..module import Module
from ..state import NeuronStates


class NeuronModel(Module):
    """Simulates dynamics of leaky-integrate-and-fire neurons in a network.

    This model consists of the connectivity between a given number of
    recurrently connected neurons as well as the update equations for currents,
    voltages, and spikes; and is used to update neuron states
    (:class:`NeuronStates`) over time.

    This model assumes that there are four neuron types:

        0. Excitatory neurons
        1. SOM neurons (inhibitory)
        2. PV neurons (inhibitory)
        3. VIP neurons (inhibitory)

    Each neuron type has different update parameters.

    Args:
        num_exc_neurons:

            The number of excitatory neurons (type 0 above).

        num_inh_neurons:

            The total number of all inhibitory neurons (types 1-3 above). SOM
            neurons will be 40% of those, and PV/VIP neurons 30% each.

        key:

            A JAX random key to initialize the neuron positions and connectivity.
    """

    num_exc_neurons: int
    num_som_neurons: int
    num_pv_neurons: int
    num_vip_neurons: int

    neuron_positions: jax.Array
    neuron_types: jax.Array

    connectivity: jax.Array

    exc_radius: jnp.float32
    inh_radius: jnp.float32

    unique_encoding_level: jnp.float32
    shared_encoding_level: jnp.float32

    tau_pv_to_exc: jnp.float32
    tau_pv_to_pv: jnp.float32
    tau_som_to_exc: jnp.float32
    tau_som_to_vip: jnp.float32
    tau_som_to_pv: jnp.float32
    tau_vip_to_som: jnp.float32

    tau_vip: jnp.float32
    tau_som: jnp.float32
    tau_pv: jnp.float32
    tau_exc: jnp.float32
    tau_membrane: jnp.float32

    def __init__(
        self,
        num_exc_neurons: int,
        num_inh_neurons: int,
        key: jax.Array,
    ):
        self.num_exc_neurons = num_exc_neurons

        # compute number of inhibitory subtype neurons
        self.num_pv_neurons = int(num_inh_neurons * 0.3)
        self.num_som_neurons = int(num_inh_neurons * 0.3)
        self.num_vip_neurons = (
            num_inh_neurons - self.num_som_neurons - self.num_pv_neurons
        )

        # randomly initialize neuron positions in 2D (on the cortical sheet)
        subkey, key = jax.random.split(key)
        self.neuron_positions = self._random_positions(subkey)

        # randomly connect neurons
        self.connectivity = self._random_connectivity(subkey)

        # assign each neuron its corresponding type for later lookups
        #   0: exc neurons
        #   1: SOM interneurons
        #   2: PV interneurons
        #   3: VIP interneurons
        self.neuron_types = jnp.array(
            [0] * self.num_exc_neurons
            + [1] * self.num_som_neurons
            + [2] * self.num_pv_neurons
            + [3] * self.num_vip_neurons,
            dtype=jnp.int32,
        )

        key, k1, k2 = jax.random.split(key, 3)
        self.exc_radius = 0.2 + 0.05 * jax.random.uniform(
            k1, shape=(num_exc_neurons, 1)
        )
        self.inh_radius = 0.15 + 0.03 * jax.random.uniform(
            k2, shape=(num_inh_neurons, 1)
        )
        # percentage of uniquely selective exc neurons
        self.unique_encoding_ratio = unique_encoding_ratio
        self.shared_encoding_ratio = shared_encoding_ratio
        (
            self.unique_encoding_cells,
            self.shared_encoding_cells,
            self.nonencoding_cells,
            key,
        ) = self._assign_encoding_selectivity(key)

        # decay of the post-synaptic excitatory currents
        self.tau_pv_to_exc = 5.0 / 1000
        self.tau_pv_to_pv = 4.6 / 1000
        self.tau_som_to_exc = 4.1 / 1000
        self.tau_som_to_vip = 10.2 / 1000
        self.tau_som_to_pv = 5.0 / 1000
        self.tau_vip_to_som = 13.1 / 1000

        self.tau_vip = 10.9 / 1000
        self.tau_som = 11.8 / 1000
        self.tau_pv = 3.1 / 1000
        self.tau_exc = 2 / 1000  # sec
        self.tau_membrane = 10.5 / 1000

        # time constant of the control signal
        self.tau_nmda_decay = 100 / 1000
        self.tau_nmda_rise = 2 / 1000

        # spiking threshold
        self.v_threshold = -50  # mV
        self.v_reset = -60
        # reversal potential V
        self.v_exc = 0
        self.v_inh = -70
        # leaky-reversal potential
        self.v_leaky = -70  # (i.e., E_L)
        # membrane capacitance
        self.c_membrane = 0.2  # nF

        self.nmda_alpha = 0.9

    @property
    def num_neurons(self):
        return (
            self.num_exc_neurons
            + self.num_pv_neurons
            + self.num_som_neurons
            + self.num_vip_neurons
        )

    @property
    def num_inh_neurons(self):
        return self.num_pv_neurons + self.num_som_neurons + self.num_vip_neurons

    def initialize_neuron_states(self, key: jax.Array) -> NeuronStates:
        """Create initial neuron states."""
        # each neuron receives input currents from 4 neuron types, set to 0 initially
        # (currents form different neuron types affect the neuron differently,
        # which is why we keep them separate)
        currents = jnp.zeros((self.num_neurons, 4), dtype=jnp.float32)

        # each neuron has a voltage, set to -60 mV initially
        voltages = -60 * jnp.ones((self.num_neurons, 4), dtype=jnp.float32)

        return NeuronStates(
            positions=positions,
            currents=currents,
            voltages=voltages,
            types=types,
        )

    def compute_activations(self, state: NeuronStates) -> jax.Array:
        """Initialize a new state of synaptic activations of all the neurons."""
        # what should happen here:
        # * we can assume that we have the current currents and voltages for
        #   each neuron, given in `state`
        # * from the voltages, we can compute whether a neuron should spike
        # * from the spikes and connectivity, we compute new internal activations
        # * return only those internal activations of shape (n,)

    def update_currents(self, state: NeuronStates, activations: jax.Array) -> jax.Array:
        # what should happen here:
        # * given all activations and the current voltages, compute the currents
        # * return only those currents (n,)

        return jax.vmap(self._update_current)(state.voltages, activations)

    def update_voltages(self, state: NeuronStates, currents: jax.Array) -> jax.Array:
        # same principle as for update_currents

        return jax.vmap(self._update_voltage)(state.voltages, currents)

    def _random_positions(self, key: jax.Array):
        return jax.random.uniform(key, shape=(self.num_neurons, 2), dtype=jnp.float32)

    def _random_connectivity(self, key: jax.Array):
        # TODO
        return jnp.ones((self.num_neurons, self.num_neurons), dtype=jnp.float32)

    def _assign_encoding_selectivity(self, key: jax.Array):
        num_unique_encoding_cell_per_item = int(
            self.unique_encoding_ratio * self.num_exc_neurons
        )
        num_unique_encoding_cells = self.num_samples * num_unique_encoding_cell_per_item
        if num_unique_encoding_cells > self.num_exc_neurons:
            raise ValueError(
                "Number of uniquely selective exc exceeds the number of exc neurons"
            )
        # front-indexed neurons are assumed to have unique encoding
        unique_encoding_cells = jnp.arange(
            num_unique_encoding_cells, dtype=jnp.int32
        ).reshape(self.num_samples, num_unique_encoding_cell_per_item)

        num_shared_encoding_cell_per_item = int(
            self.shared_encoding_ratio * self.num_exc_neurons
        )
        num_shared_encoding_cells = self.num_samples * num_shared_encoding_cell_per_item
        if num_shared_encoding_cells > self.num_exc_neurons:
            raise ValueError(
                "Number of uniquely selective exc exceeds the number of exc neurons"
            )
        # sample from the remaining exc neurons to have shared encodings
        # renaming for readability
        total_used = num_unique_encoding_cells
        k = num_unique_encoding_cell_per_item
        shifted_indices = jnp.arange(total_used - k, dtype=jnp.int32)

        def sample(i, subkey):
            start = jnp.int32(i * num_unique_encoding_cell_per_item)
            # For each item i, "remaining" indices are [0, total_used)
            # excluding [i*k, (i+1)*k).
            remaining = shifted_indices + jnp.where(
                shifted_indices >= start, jnp.int32(k), jnp.int32(0)
            )
            # sample without replacement by permutation
            chosen = jax.random.permutation(subkey, remaining)[
                :num_shared_encoding_cell_per_item
            ]
            return chosen

        keys = jax.random.split(key, self.num_samples + 1)
        key_out, item_keys = keys[0], keys[1:]
        # shared_encoding: (num_samples, num_shared_encoding_cell_per_item)
        shared_encoding_cells = jax.vmap(sample)(
            jnp.arange(self.num_samples, dtype=jnp.int32), item_keys
        )
        # under this construction, the nonselective E-cells are exactly the
        # "unused tail"
        nonencoding_cells = jnp.arange(num_unique_encoding_cells, self.num_exc_neurons)

        return unique_encoding_cells, shared_encoding_cells, nonencoding_cells, key_out

    def _update_firing(self, state: NeuronStates):
        """Zhen: Could take this out for now."""
        pass

    def _update_current(self, voltage, activation) -> jnp.float32:
        # here goes the update formula, all needed parameters should already be in "self"
        return 0.0

    def _update_voltage(self, voltage, current) -> jnp.float32:
        # here goes the update formula, all needed parameters should already be in "self"
        return 0.0
