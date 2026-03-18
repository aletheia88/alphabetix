import jax
import jax.numpy as jnp

from ..module import Module


class Timeline(Module):
    num_samples: int
    cue_length: int
    delay_length: int
    dt: float
    total_length: float
    num_timesteps: int
    timesteps: jax.Array
    timestep_to_cue: jax.Array

    def __init__(
        self,
        num_samples: int,
        cue_length: int,
        delay_length: int,
        dt: float,
    ):
        self.dt = dt
        self.total_length = (cue_length + delay_length) * num_samples
        self.num_timesteps = round(self.total_length / dt)
        self.cue_timesteps = round(self.cue_length / dt)
        self.timesteps = jnp.arange(self.num_timesteps, dtype=jnp.float32)

        unit_length = cue_length + delay_length
        num_unit_timesteps = round(unit_length / dt)
        self.timestep_to_cue = (self.timesteps // num_unit_timesteps).astype(jnp.int32)

        cue_indices = jnp.arange(num_samples, dtype=jnp.int32)
        self.start_timesteps = jnp.asarray(
            jnp.round(cue_indices * unit_length / dt), dtype=jnp.int32
        )

    def cue_index(self, timestep: int) -> jax.Array:
        """Map from timestep to cue / sample index."""
        return self.timestep_to_cue[jnp.int32(timestep)]
