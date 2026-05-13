from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass
class Timeline:
    """Timeline for structured working-memory tasks with time-indexed event lookup.

    This class represents a task in which a sequence of items is presented over time,
    interleaved with delay periods. It provides a fast lookup from a given time `t`
    to an "event" representation consisting of:
        (1) a temporal encoding (position in the sequence)
        (2) a category encoding (identity of the item)

    (1) Temporal Encoding
    ---------------------
    Encodes the position of a cue in the sequence.

        - Dimension: num_cues = num_categories + 1
        - Index 0: always corresponds to a delay period
        - Indices 1..N: correspond to the 1st, 2nd, ..., N-th cue

    Example (sequence = "cab"):
        delay → (1, 0, 0, 0)
        c     → (0, 1, 0, 0)
        a     → (0, 0, 1, 0)
        b     → (0, 0, 0, 1)

    (2) Category Encoding
    ---------------------
    Encodes the identity of the item.

        - Dimension: num_categories = number of unique items in `sequence`
        - Categories are assigned indices in alphabetical order
        - Delay periods map to the zero vector

    Example (sequence = "cab"):
        a → (1, 0, 0)
        b → (0, 1, 0)
        c → (0, 0, 1)
        delay → (0, 0, 0)
    """

    cue_time: int
    delay_time: int
    sequence: str

    num_categories: int = field(init=False)
    num_cues: int = field(init=False)
    categories: tuple[str, ...] = field(init=False)
    category_to_index: dict[str, int] = field(init=False)

    segment_labels: tuple[str, ...] = field(init=False)
    segment_starts: jnp.ndarray = field(init=False)
    segment_ends: jnp.ndarray = field(init=False)

    temporal_table: jnp.ndarray = field(init=False)
    category_table: jnp.ndarray = field(init=False)

    total_time: int = field(init=False)

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("`sequence` must be non-empty.")
        if self.cue_time <= 0 or self.delay_time <= 0:
            raise ValueError("`cue_time` and `delay_time` must be positive.")

        self.categories = tuple(sorted(set(self.sequence)))
        self.num_categories = len(self.categories)
        self.num_cues = len(self.sequence) + 1
        self.category_to_index = {
            category: i for i, category in enumerate(self.categories)
        }

        labels = []
        starts = []
        ends = []

        t = 0

        for i, item in enumerate(self.sequence):
            labels.append("delay")
            starts.append(t)
            t += self.delay_time
            ends.append(t)

            labels.append(item)
            starts.append(t)
            t += self.cue_time
            ends.append(t)

        self.segment_labels = tuple(labels)
        self.segment_starts = jnp.array(starts)
        self.segment_ends = jnp.array(ends)
        self.total_time = t

        eye_temporal = jnp.eye(self.num_cues, dtype=jnp.float32)
        eye_category = jnp.eye(self.num_categories, dtype=jnp.float32)

        temporal_rows = []
        category_rows = []

        cue_position = 0

        for label in self.segment_labels:
            if label == "delay":
                temporal_rows.append(eye_temporal[0])
                category_rows.append(
                    jnp.zeros((self.num_categories,), dtype=jnp.float32)
                )
            else:
                cue_position += 1
                temporal_rows.append(eye_temporal[cue_position])
                category_rows.append(eye_category[self.category_to_index[label]])

        self.temporal_table = jnp.stack(temporal_rows)
        self.category_table = jnp.stack(category_rows)

    def lookup_index(self, t: int) -> jnp.ndarray:
        if t < 0 or t >= self.total_time:
            raise ValueError(f"`t={t}` is outside [0, {self.total_time}).")

        mask = (t >= self.segment_starts) & (t < self.segment_ends)
        return jnp.argmax(mask)

    def lookup_temporal(self, t: int) -> jnp.ndarray:
        return self.temporal_table[self.lookup_index(t)]

    def lookup_category(self, t: int) -> jnp.ndarray:
        return self.category_table[self.lookup_index(t)]

    def lookup_label(self, t: int) -> str:
        return self.segment_labels[int(self.lookup_index(t))]

