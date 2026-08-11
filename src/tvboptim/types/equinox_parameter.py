"""Explicit trainable-parameter marker for Equinox modules."""

import equinox as eqx


class EquinoxParameter(eqx.Module):
    """Mark an Equinox module's inexact array leaves as trainable.

    ``Space`` does not support axes inside an ``EquinoxParameter``. PyTree-visible
    axes are rejected explicitly; fields declared with ``eqx.field(static=True)``
    are preserved as static module metadata and are not inspected or swept. Place
    all ``Space`` axes outside the module.
    """

    module: eqx.Module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
