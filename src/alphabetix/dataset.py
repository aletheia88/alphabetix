from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .models.timeline import Timeline, TimelineInputs


@dataclass(frozen=True)
class TimelineDataset:
    """Stack encodings from multiple timelines."""

    timelines: tuple[Timeline, ...]
    inputs: TimelineInputs

    @classmethod
    def from_timelines(
        cls,
        timelines: Sequence[Timeline],
        dt: float,
    ) -> "TimelineDataset":
        timelines = tuple(timelines)
        if not timelines:
            raise ValueError("A TimelineDataset must contain at least one timeline.")

        # collect encodings over all timesteps for all timelines
        all_timeline_encodings = tuple(
            timeline.get_inputs(dt) for timeline in timelines
        )
        # stack all encodings:
        # temporal encodings: (num_examples, T, num_cues)
        # category encodings: (num_examples, T, num_categories)
        inputs = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves),
            *all_timeline_encodings,
        )
        return cls(timelines=timelines, inputs=inputs)

    def __len__(self) -> int:
        return len(self.timelines)

    def __getitem__(self, index: int):
        """Return the temporal and category encodings for one timeline.

        e.g., dataset[7] -> TimelineInputs(
        temporal_encodings=...,  # (T, num_cues)
        category_encodings=...,  # (T, num_categories))

        e.g., dataset.timelines[7].sequence
        """
        return jax.tree_util.tree_map(lambda leaf: leaf[index], self.inputs)

    @property
    def num_timesteps(self) -> int:
        return self.inputs.temporal_encodings.shape[1]

    @property
    def num_cues(self) -> int:
        return self.inputs.temporal_encodings.shape[2]

    @property
    def num_categories(self) -> int:
        return self.inputs.category_encodings.shape[2]
