import equinox as eqx
import jax
import jax.numpy as jnp

from ..module import Module


class SensoryModelMLP(Module):
    layer: eqx.nn.Linear
    num_categories: int = Module.static()
    num_neurons: int = Module.static()

    def __init__(
        self,
        num_categories: int,
        num_neurons: int,
        *,
        key: jax.Array,
    ):
        self.num_categories = num_categories
        self.num_neurons = num_neurons
        self.layer = eqx.nn.Linear(num_categories, num_neurons, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim != 1:
            raise ValueError(
                f"SensoryModel expects a 1D input of shape ({self.num_categories},), "
                f"got shape {x.shape}."
            )
        if x.shape[0] != self.num_categories:
            raise ValueError(
                f"Expected input dimension {self.num_categories}, got {x.shape[0]}."
            )

        return self.layer(x)


class SensoryModel(Module):
    coding_matrix: jax.Array
    # unique_assignment: (num_neurons,)
    # neurons with non-unique coding is assigned -1
    unique_assignment: jax.Array
    coding_mask: jax.Array
    shared_mask: jax.Array

    num_categories: int = Module.static()
    num_neurons: int = Module.static()
    percent_coding: float = Module.static()
    percent_unique_coding: float = Module.static()
    percent_shared_coding: float = Module.static()
    stim_amplitude: float = Module.static()

    num_coding: int = Module.static()
    num_unique_coding: int = Module.static()
    num_shared_coding: int = Module.static()

    def __init__(
        self,
        num_categories: int,
        num_neurons: int,
        percent_coding: float,
        percent_unique_coding: float,
        percent_shared_coding: float,
        stim_amplitude: float,
        *,
        key: jax.Array,
    ):
        self.num_categories = num_categories
        self.num_neurons = num_neurons
        self.percent_coding = percent_coding
        self.percent_unique_coding = percent_unique_coding
        self.percent_shared_coding = percent_shared_coding
        self.stim_amplitude = stim_amplitude

        self.num_coding = int(num_neurons * percent_coding + 0.5)
        self.num_unique_coding = int(self.num_coding * percent_unique_coding + 0.5)
        self.num_shared_coding = self.num_coding - self.num_unique_coding

        if percent_coding > 0.0 and self.num_coding == 0:
            raise ValueError(
                "`percent_coding` is positive but rounds to zero coding "
                "neurons; increase it or `num_neurons`."
            )
        if (
            self.num_coding > 0
            and percent_unique_coding > 0.0
            and (self.num_unique_coding < self.num_categories)
        ):
            raise ValueError(
                "There are too few unique coding neurons to give every "
                "category at least one. Increase `num_neurons`, "
                "`percent_coding`, or `percent_unique_coding`."
            )
        if self.num_shared_coding > 0 and self.num_categories < 2:
            raise ValueError("Shared coding requires at least two categories.")

        coding_key, category_key = jax.random.split(key)
        neuron_order = jax.random.permutation(coding_key, num_neurons)
        coding_indices = neuron_order[: self.num_coding]
        unique_indices = coding_indices[: self.num_unique_coding]
        shared_indices = coding_indices[self.num_unique_coding :]

        category_order = jax.random.permutation(category_key, num_categories)
        per_category, remainder = divmod(
            self.num_unique_coding,
            self.num_categories,
        )
        coding_matrix = jnp.zeros(
            (num_categories, num_neurons),
            dtype=jnp.bool_,
        )
        unique_assignment = jnp.full((num_neurons,), -1, dtype=jnp.int32)

        start = 0
        for position in range(num_categories):
            category = category_order[position]
            count = per_category + int(position < remainder)
            category_indices = unique_indices[start : start + count]

            coding_matrix = coding_matrix.at[category, category_indices].set(True)
            unique_assignment = unique_assignment.at[category_indices].set(category)
            start += count

        coding_matrix = coding_matrix.at[:, shared_indices].set(True)

        self.coding_matrix = coding_matrix
        self.unique_assignment = unique_assignment
        self.coding_mask = (
            jnp.zeros((num_neurons,), dtype=jnp.bool_).at[coding_indices].set(True)
        )
        self.shared_mask = (
            jnp.zeros((num_neurons,), dtype=jnp.bool_).at[shared_indices].set(True)
        )

    def __call__(self, category_encoding: jax.Array) -> jax.Array:
        return self.stim_amplitude * (
            category_encoding @ self.coding_matrix.astype(jnp.float32)
        )
