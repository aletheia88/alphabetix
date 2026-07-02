import dataclasses

import equinox as eqx
import jax


class Module(eqx.Module):
    def param(**kwargs):
        return eqx.field(
            metadata={"partition": "param"},
            **kwargs,
        )

    def static(**kwargs):
        return eqx.field(
            metadata={"partition": "static"},
            **kwargs,
        )

    def replace(
        self,
        **kwargs,
    ):
        return eqx.tree_at(
            lambda c: tuple(getattr(c, name) for name in kwargs.keys()),
            self,
            tuple(kwargs.values()),
        )

    def partition(self):
        return eqx.partition(self, self._filter())

    def _filter(self):
        """Split into trainable and non-trainable values based on fields' metadata."""
        leaves = {}

        for field in dataclasses.fields(self):
            print(field.name)
            value = getattr(self, field.name)
            field_type = field.metadata.get("partition", "static")

            if isinstance(value, Module):
                # for our modules, we call this filter recursively
                leaves[field.name] = value._filter()
            elif isinstance(value, eqx.Module):
                # for equinox modules, we go with the default partition:
                # mark a leaf `True` (i.e., trainable) if it is an array
                leaves[field.name] = jax.tree_util.tree_map(eqx.is_array, value)
            else:
                # for all other fields, we only accept those with metatdata
                # "partition" set to "param" as parameters
                leaves[field.name] = field_type == "param"

        return eqx.tree_at(
            lambda c: tuple(getattr(c, name) for name in leaves.keys()),
            self,
            tuple(leaves.values()),
        )
