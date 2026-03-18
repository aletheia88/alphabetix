import jax
import jax.numpy as jnp

from .neuron_model import NeuronModel
from .timeline import Timeline


class InputModel(Timeline):
    mu_exc: jnp.float32
    sigma_exc: jnp.float32
    mu_pv: jnp.float32
    sigma_pv: jnp.float32
    mu_som: jnp.float32
    sigma_som: jnp.float32
    mu_vip: jnp.float32
    sigma_vip: jnp.float32
    tau: jnp.float32

    tau_nmda_decay: jnp.float32
    tau_nmda_rise: jnp.float32
    nmda_alpha: jnp.float32
    tau_exc: jnp.float32

    num_thalam_fibers: int

    neuron_model: NeuronModel
    num_neurons: int
    feedforward_connectivity: jax.Array
    feedback_connectivity: jax.Array
    cortex_connectivity: jax.Array
    # input spikes
    cortex_spikes: jax.Array
    thalam_spikes: jax.Array
    nmda_spikes: jax.Array

    def __init__(self, neuron_model: NeuronModel, key: jax.Array):
        # unique_encoding_ratio: float,
        # shared_encoding_ratio: float,
        # neurophysiological constants for cortical cell types
        self.mu_exc = 200
        self.sigma_exc = 5
        self.mu_pv = 770
        self.sigma_pv = 10
        self.mu_som = 220
        self.sigma_som = 50
        self.mu_vip = 150
        self.sigma_vip = 30
        self.tau = 5 / 1000
        # neurophysiological constants for NMDA-mediated presynaptic neurons
        self.tau_nmda_decay = 100 / 1000
        self.tau_nmda_rise = 2 / 1000
        self.nmda_alpha = 0.9

        self.neuron_model = neuron_model
        self.num_neurons = neuron_model.num_neurons
        self.tau_exc = neuron_model.tau_exc

        # connectivity matrices
        self.feedforward_connectivity = jax.Array([])
        self.feedback_connectivity = jax.Array([])
        self.cortex_connectivity = jax.Array([])

        self.num_thalam_fibers = 100

        # generating incoming spikes from cortex
        key1, key2, key3, key4, key5, key6, key = jax.random.split(key, 7)
        vip_spikes = self._generate_spikes(
            key,
            neuron_model.num_vip_neurons,
            self.mu_vip,
            self.sigma_vip,
            self.tau,
        )
        som_spikes = self._generate_spikes(
            key1,
            neuron_model.num_som_neurons,
            self.mu_som,
            self.sigma_som,
            self.tau,
        )
        pv_spikes = self._generate_spikes(
            key2,
            neuron_model.num_pv_neurons,
            self.mu_pv,
            self.sigma_pv,
            self.tau,
        )
        exc_spikes = self._generate_spikes(
            key3,
            neuron_model.num_exc_neurons,
            self.mu_exc,
            self.sigma_exc,
            self.tau,
        )
        self.cortex_spikes = jnp.vstack([exc_spikes, som_spikes, pv_spikes, vip_spikes])

        # generating incoming spikes from thalamus
        spike_rate_thalam_exc = jnp.float32(50.0) + jnp.float32(
            70.0
        ) * jax.random.uniform(key4, (self.num_thalam_fibers, self.num_timesteps))
        self.thalam_spikes = self._poisson(
            key5,
            spike_rate_thalam_exc,
            self.num_thalam_fibers,
        )

        # generating incoming spikes mediated by NMDA receptors
        spike_rate_nmda = 25.0
        self.nmda_spikes = self._poisson(key6, spike_rate_nmda, self.num_thalam_fibers)

        # self.spikes = jnp.concatenate(
        #     [self.cortex_spikes, self.thalam_spikes, self.nmda_spikes]
        # ).T

        # TODO: generating stimulus simulating the effect of input sensory data
        self.input_stimulus = self._generate_stimuli()

    def compute_activations(self):
        """Pre-compute all input activations for all timesteps here return as (t, n) array."""
        # initialize synaptic activations due to general cortical activity
        initial_cortex_activations = jnp.zeros(
            (self.num_timesteps, self.num_neurons), dtype=jnp.float32
        )
        initial_thalam_activations = jnp.zeros(
            (self.num_timesteps, self.num_neurons), dtype=jnp.float32
        )
        initial_nmda_activations = jnp.zeros(
            (self.num_timesteps, self.num_neurons), dtype=jnp.float32
        )
        initial_nmda_x = jnp.zeros(
            (self.num_timesteps, self.num_neurons), dtype=jnp.float32
        )

        def step_fn(carry):
            cortex_activations, thalam_activations, nmda_activations, nmda_x = carry
            next_cortex_activations = (
                cortex_activations
                - (self.dt / self.tau_exc) * cortex_activations
                + self.cortex_spikes * self.cortex_connectivity
            )

            active_thalam_fibers = (self.thalam_spikes != 0).astype(jnp.float32)
            thalam_fiber_weights = self.feedforward_connectivity @ active_thalam_fibers
            next_thalam_activations = (
                thalam_activations
                - (self.dt / self.tau_exc) * thalam_activations
                + thalam_fiber_weights * self.input_stimulus
            )

            next_nmda_activations = (
                nmda_activations
                - (self.dt / self.tau_nmda_decay) * nmda_activations
                + self.dt * self.nmda_alpha * nmda_x * (1.0 - nmda_activations)
            )
            active_nmda_fibers = (self.nmda_spikes != 0).astype(jnp.float32)
            nmda_fiber_weights = self.feedback_connectivity @ active_nmda_fibers
            nmda_drive = jnp.concatenate(
                [
                    jnp.zeros((self.neuron_model.num_exc_neurons,), dtype=jnp.float32),
                    nmda_fiber_weights,
                ]
            )
            next_nmda_x = nmda_x - (self.dt / self.tau_nmda_rise) * nmda_x + nmda_drive

            next_carry = (
                next_cortex_activations,
                next_thalam_activations,
                next_nmda_activations,
                next_nmda_x,
            )
            y = (
                next_cortex_activations,
                next_thalam_activations,
                next_nmda_activations,
                next_nmda_x,
            )
            return next_carry, y

        (_, _), (cortex_activations, thalam_activations, nmda_activations, nmda_x) = (
            jax.lax.scan(
                step_fn,
                (
                    initial_cortex_activations,
                    initial_thalam_activations,
                    initial_nmda_activations,
                    initial_nmda_x,
                ),
            )
        )
        return cortex_activations, thalam_activations, nmda_activations, nmda_x

    def _generate_spikes(
        self,
        key,
        num_cells,
        mu,
        sigma,
        tau,
    ):
        # sampling x from the Ornstein-Ulenbeck stochastic process
        mu = jnp.asarray(mu)
        x0 = jnp.broadcast_to(mu, (num_cells,))  # works if mu is scalar or (num_cells,)

        def step(carry, _):
            x, key = carry
            key, subkey = jax.random.split(key)

            d_drift = (mu - x) * self.dt
            d_wiener = jnp.sqrt(self.dt * 2.0 * tau) * jax.random.normal(
                subkey, (num_cells,)
            )
            dx = (1.0 / tau) * (d_drift + sigma * d_wiener)

            x_next = x + dx
            return (x_next, key), x_next

        # xs is (self.num_timesteps, num_cells) holding x1..x_self.num_timesteps
        (_, key1), xs = jax.lax.scan(
            step, (x0, key), xs=None, length=self.num_timesteps
        )
        x_all = jnp.concatenate(
            [x0[None, :], xs], axis=0
        ).T  # (num_cells, self.num_timesteps+1)

        # sampling spikes from the Poisson process
        # spike_rate: (num_cells, self.num_timesteps)
        spike_rate = jnp.maximum(0.0, x_all[:, 1:])
        spikes = self._poisson(key1, spike_rate, num_cells)

        return spikes

    def _poisson(self, key, spike_rate, num_cells):
        probabilities = spike_rate * self.dt

        if spike_rate.ndim == 0:
            shape = (num_cells, self.num_timesteps)
            sample_probabilities = jnp.clip(probabilities, 0.0, 1.0)  # scalar
        else:
            if num_cells is not None and spike_rate.ndim == 1:
                probabilities = jnp.broadcast_to(
                    probabilities, (num_cells, spike_rate.shape[0])
                )
            shape = None
            sample_probabilities = jnp.clip(probabilities, 0.0, 1.0)

        key, subkey = jax.random.split(key)
        spikes = jax.random.bernoulli(
            subkey, p=sample_probabilities, shape=shape
        ).astype(jnp.int32)

        return spikes

    def _generate_stimuli(self):
        encoding_exc_neurons = jnp.concatenate(
            [
                self.neuron_model.unique_encoding_cells,  # (num_samples, x)
                self.neuron_model.shared_encoding_cells,  # (num_samples, y)
            ],
            axis=1,
        )
        som_cells = ...
        vip_cells = ...

        sensory_stimulus = self._build_sensory_stimulus(self.cue_timesteps)
        stimulus_train = jnp.zeros(
            (self.num_neurons, self.num_timesteps), dtype=jnp.float32
        )

        def step_fn(carry, cue_index):
            stimulus_train, key = carry

        return

    def _build_sensory_stimulus(self, l, dtype=jnp.float32):
        l1 = round(l / 4)
        l2 = round(4 * l / 5)

        head = (
            jnp.linspace(jnp.array(0.0, dtype), jnp.array(1.0, dtype), l1)
            if l1 > 0
            else jnp.empty((0,), dtype)
        )
        mid = jnp.ones((max(l2 - l1, 0),), dtype=dtype)
        tail = (
            jnp.linspace(jnp.array(1.0, dtype), jnp.array(0.8, dtype), max(l - l2, 0))
            if (l - l2) > 0
            else jnp.empty((0,), dtype)
        )
        out = jnp.concatenate([head, mid, tail], axis=0)
        # guard against rounding mismatch
        out = jax.lax.cond(
            out.shape[0] == l,
            lambda x: x,
            lambda _: jnp.ones((l,), dtype=dtype),
            out,
        )
        return out
