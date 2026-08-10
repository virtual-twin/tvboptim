"""Explicit trainable-parameter marker for Equinox modules."""

import equinox as eqx


class EquinoxParameter(eqx.Module):
    """Mark an Equinox module's inexact array leaves as trainable."""

    module: eqx.Module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
