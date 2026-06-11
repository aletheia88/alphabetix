import equinox as eqx
import jax
import jax.numpy as jnp

from spynal.spikes import rate
from spynal.spectra import wavelet, one_over_f_norm


class Electrode(eqx.Module):
    position: jax.Array  # shape: (2,)
    measurement_radius: jnp.float32

    def measure_lfp(self, neurons):
        # neuron_positions: (num_neurons, 2)
        neuron_positions = neurons.position[0, :, :]
        distances = jnp.linalg.norm(
            neuron_positions - self.position[None, :],
            axis=1,
        )
        measured_neurons = (distances < self.measurement_radius) & (
            neurons.sign[0, :] > 0
        )
        num_measured_neurons = jnp.sum(measured_neurons)
        # LFP-proxy per timestep:
        # the mean voltage at timestep t among excitatory neurons located
        # within the electrode's measurement radius
        lfp_readout = (
            jnp.sum(
                neurons.voltage * measured_neurons[None, :],
                axis=1,
            )
            / num_measured_neurons
        )

        return lfp_readout, num_measured_neurons

    def measure_single_neuron_spike_rates(self, neurons, dt):
        # NOTE: in our simulation, information of single neurons (e.g.,
        # spiking) does not need to be attained through the sorting algorithm
        # using the electrode measurements, since we just know;
        # that means, this function does not have to be under the `Electrode`
        # class
        num_timesteps = neurons.spike.shape[0]
        timepoints = jnp.arange(num_timesteps) * dt / 1000.0
        # rate_times: sec
        rates, rate_times = rate(
            neurons.spike.astype(bool),
            method="density",
            lims=[timepoints[0], timepoints[-1]],
            kernel="gaussian",
            width=50e-3,
            step=10e-3,
            timepts=timepoints,
            axis=0,
        )
        # TODO: perhaps get population firing rates from averaging ensemble of
        # single neurons?
        return rates, rate_times

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
