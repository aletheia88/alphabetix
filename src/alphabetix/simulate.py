import jax

from .model import InputModel, NeuronModel
from .sensor import Sensor
from .state import NeuronStates


def run_simulation(
    neuron_model: NeuronModel,
    sensor: Sensor,
    key: jax.Array,
):
    key, subkey = jax.random.split(key)
    initial_neuron_states = neuron_model.initialize_neuron_states(subkey)
    # NOTE: initialize timeline under neuron_model?

    # generate external inputs from cortex and thalamus
    key, subkey = jax.random.split(key)
    input_model = InputModel(neuron_model, subkey)

    input_activations = input_model.compute_activations()
    # input_activations: (t, n), bool
    # t = timesteps
    # n = number of internal neurons

    def scan_fn(neuron_state, input_activation):
        neuron_state, readout = simulation_step(
            neuron_model, neuron_state, input_activation, sensor
        )
        return neuron_state, (neuron_state, readout)

    _, (neuron_states, readouts) = jax.lax.scan(
        scan_fn, initial_neuron_states, input_activations
    )

    return neuron_states, readouts


def simulation_step(
    neuron_model: NeuronModel,
    neuron_state: NeuronStates,
    input_activations: jax.Array,
    sensor: Sensor,
) -> tuple[NeuronStates, jax.Array]:
    """Update the internal neuron states and produce sensor readings."""
    # 1. compute internal activations (n,)
    internal_activations = neuron_model.compute_activations(neuron_state)

    # 2. add to input activations (n,)
    total_activations = internal_activations + input_activations

    # 3. compute next currents and voltages
    currents = neuron_model.update_currents(neuron_state, total_activations)
    voltages = neuron_model.update_voltages(neuron_state, currents)

    # 4. update the neuron state
    neuron_state = neuron_state.replace(currents=currents, voltages=voltages)

    # 5. generate measurements
    readout = sensor.measure(neuron_state)

    return neuron_state, readout
