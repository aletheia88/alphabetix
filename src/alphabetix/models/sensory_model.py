import equinox as eqx
import jax
import jax.numpy as jnp

from ..module import Module


class SensoryModel(Module):
    layer: eqx.nn.Linear
    sensory_token_size: int = Module.static()
    num_neurons: int = Module.static()

    def __init__(
        self,
        vocabulary_size: int,
        num_neurons: int,
        *,
        key: jax.Array,
    ):
        # plus 1 comes from the dimension representing empty information during
        # the prestim and delay periods
        self.sensory_token_size = vocabulary_size + 1
        self.num_neurons = num_neurons
        self.layer = eqx.nn.Linear(
            self.sensory_token_size,
            self.num_neurons,
            use_bias=False,
            key=key,
        )

    def __call__(self, category_encoding: jax.Array) -> jax.Array:
        if category_encoding.shape != (self.sensory_token_size,):
            raise ValueError(
                f"Expected sensory encoding shape {(self.sensory_token_size,)}, "
                f"got {category_encoding.shape}."
            )

        return self.layer(category_encoding)


class ExplicitSensoryModel(SensoryModel):
    """Learn category-specific currents through a fixed, presampled coding mask."""

    coding_matrix: jax.Array
    # unique_assignment: (num_neurons,)
    # neurons with non-unique coding is assigned -1
    unique_assignment: jax.Array
    coding_mask: jax.Array
    shared_mask: jax.Array

    vocabulary_size: int = Module.static()

    num_coding: int = Module.static()
    num_unique_coding: int = Module.static()
    num_shared_coding: int = Module.static()

    percent_coding: float = Module.static(default=0.3)
    percent_unique_coding: float = Module.static(default=0.7)
    percent_shared_coding: float = Module.static(default=0.3)

    def __init__(
        self,
        vocabulary_size: int,
        num_neurons: int,
        percent_coding: float = 0.3,
        percent_unique_coding: float = 0.7,
        percent_shared_coding: float = 0.3,
        *,
        key: jax.Array,
    ):
        if (
            not all(
                0 <= value <= 1
                for value in (
                    percent_coding,
                    percent_unique_coding,
                    percent_shared_coding,
                )
            )
            or abs(percent_unique_coding + percent_shared_coding - 1) > 1e-6
        ):
            raise ValueError(
                "Coding percentages must be in [0, 1]; unique and shared must sum to 1."
            )
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive.")

        coding_key, category_key, weight_key = jax.random.split(key, 3)

        super().__init__(vocabulary_size, num_neurons, key=weight_key)

        self.vocabulary_size = vocabulary_size
        self.percent_coding = percent_coding
        self.percent_unique_coding = percent_unique_coding
        self.percent_shared_coding = percent_shared_coding

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
            and (self.num_unique_coding < self.vocabulary_size)
        ):
            raise ValueError(
                "There are too few unique coding neurons to give every "
                "category at least one. Increase `num_neurons`, "
                "`percent_coding`, or `percent_unique_coding`."
            )
        if self.num_shared_coding > 0 and self.vocabulary_size < 2:
            raise ValueError("Shared coding requires at least two categories.")

        neuron_order = jax.random.permutation(coding_key, num_neurons)
        coding_indices = neuron_order[: self.num_coding]
        unique_indices = coding_indices[: self.num_unique_coding]
        shared_indices = coding_indices[self.num_unique_coding :]

        category_order = jax.random.permutation(category_key, vocabulary_size)
        per_category, remainder = divmod(
            self.num_unique_coding,
            self.vocabulary_size,
        )
        coding_matrix = jnp.zeros(
            (self.sensory_token_size, num_neurons),
            dtype=jnp.bool_,
        )
        # category index for neurons with unique encoding, -1 otherwise
        unique_assignment = jnp.full((num_neurons,), -1, dtype=jnp.int32)

        start = 0
        for position in range(vocabulary_size):
            category = category_order[position]
            count = per_category + int(position < remainder)
            category_indices = unique_indices[start : start + count]

            coding_matrix = coding_matrix.at[category, category_indices].set(True)
            unique_assignment = unique_assignment.at[category_indices].set(category)
            start += count

        coding_matrix = coding_matrix.at[:vocabulary_size, shared_indices].set(True)

        self.coding_matrix = coding_matrix
        self.unique_assignment = unique_assignment
        self.coding_mask = (
            jnp.zeros((num_neurons,), dtype=jnp.bool_).at[coding_indices].set(True)
        )
        self.shared_mask = (
            jnp.zeros((num_neurons,), dtype=jnp.bool_).at[shared_indices].set(True)
        )

    def __call__(self, category_encoding: jax.Array) -> jax.Array:
        if category_encoding.shape != (self.sensory_token_size,):
            raise ValueError(
                f"Expected sensory encoding shape {(self.sensory_token_size,)}."
            )
        weights = self.layer.weight * self.coding_matrix.T
        return weights @ category_encoding
