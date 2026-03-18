import equinox as eqx


class Module(eqx.Module):
    def replace(
        self,
        **kwargs,
    ):
        return eqx.tree_at(
            lambda c: tuple(getattr(c, name) for name in kwargs.keys()),
            self,
            tuple(kwargs.values()),
        )
