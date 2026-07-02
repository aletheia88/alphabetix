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
