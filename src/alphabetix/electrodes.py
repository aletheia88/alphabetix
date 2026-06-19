import equinox as eqx
import jax
import jax.numpy as jnp

from spynal.spikes import rate
from spynal.spectra import wavelet, one_over_f_norm


class Electrode(eqx.Module):
    position: jax.Array  # shape: (2,)
    measurement_radius: jnp.float32

    def __init__(self, position, measurement_radius):
        self.position = position
        self.measurement_radius = measurement_radius

    def count_measured_neurons(self, measured_neurons):
        return jnp.sum(measured_neurons)

    def measure_lfp(self, neurons):
        measured_exc_neurons = self._find_neurons(neurons, neuron_type=1)
        num_measured_exc_neurons = self.count_measured_neurons(measured_exc_neurons)
        return (
            jnp.sum(
                neurons.voltage * measured_exc_neurons[None, :],
                axis=1,
            )
            / num_measured_exc_neurons
        )

    def measure_population_firing_rate(self, neurons, neuron_type, dt):
        firing_rates, rate_timepoints = self._single_neuron_firing_rates(neurons, dt)

        measured_neurons = self._find_neurons(neurons, neuron_type)
        num_measured_neurons = self.count_measured_neurons(measured_neurons)
        return (
            jnp.sum(
                firing_rates * measured_neurons[None, :],
                axis=1,
            )
            / num_measured_neurons
        ), rate_timepoints

    def measure_wavelet_spectrogram(self, neurons, rhythm, dt):
        sampling_rate = 1000.0 / dt

        if rhythm == "alpha/beta":
            rhythm_frequencies = jnp.arange(8, 30, 2)

        if rhythm == "gamma":
            rhythm_frequencies = jnp.arange(30, 150, 2)

        # steps to compute spectrogram
        # 1. measure LFP (remove evoked potentials if num_trials > 0)
        # 2. call `wave_spectrogram`
        # 3. normalized spectrum by multiplying frequency (since high freq -> low power)

        # lfp: (num_timesteps,)
        lfp, _ = self.measure_lfp(neurons)

        # spectra: (num_frequency_bands, num_output_timepoints)
        spectrum, frequencies, spectrum_timepoints = wavelet.wavelet_spectrogram(
            lfp,
            smp_rate=sampling_rate,
            axis=0,
            data_type="spike",
            spec_type="power",
            freqs=rhythm_frequencies,
            wavelet="morlet",
            wavenumber=6,
            removeDC=True,
            downsmp=100,
        )
        spectrum_normalized = one_over_f_norm(
            spectrum,
            axis=0,
            freqs=frequencies,
        )

        return spectrum_normalized, frequencies, spectrum_timepoints

    def _single_neuron_firing_rates(self, neurons, dt):
        num_timesteps = neurons.spike.shape[0]
        timepoints = jnp.arange(num_timesteps) * dt / 1000.0
        # rate_timepoints: sec
        rates, rate_timepoints = rate(
            neurons.spike.astype(bool),
            method="density",
            lims=[timepoints[0], timepoints[-1]],
            kernel="gaussian",
            width=50e-3,
            step=10e-3,
            timepts=timepoints,
            axis=0,
        )
        return rates, rate_timepoints

    def _find_neurons(self, neurons, neuron_type):
        positions = neurons.position
        if positions.ndim == 2:
            neuron_positions = positions
        elif positions.ndim == 3:
            # neuron_positions: (num_timesteps, num_neurons, 2)
            neuron_positions = neurons.position[0, :, :]

        distances = jnp.linalg.norm(
            neuron_positions - self.position[None, :],
            axis=1,
        )
        return (distances < self.measurement_radius) & (
            neurons.type[0, :] == neuron_type
        )
