"""Abstract base class for external inputs.

This module defines the interface for external inputs that can be added to
network dynamics simulations. External inputs are time-dependent (and optionally
state-dependent) signals that influence node dynamics independently of network
coupling.

The interface mirrors the coupling system with prepare/compute/update_state
pattern, supporting both stateless and stateful inputs.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp

from ..core.bunch import Bunch


class AbstractExternalInput(ABC):
    """Abstract base class for external inputs.

    External inputs are functions of time (and optionally state) that provide
    additional drive to node dynamics, independent of network coupling.

    Like couplings, external inputs follow a three-stage pattern:
    1. prepare(): Setup and precomputation before simulation
    2. compute(): Calculate input at each timestep during simulation
    3. update_state(): Update internal state after each integration step

    Class Attributes:
        N_OUTPUT_DIMS: Number of output dimensions
        DEFAULT_PARAMS: Default parameter values as Bunch

    Instance Attributes:
        params: Instance-specific parameters (copy of DEFAULT_PARAMS + overrides)
    """

    N_OUTPUT_DIMS: int = 1
    DEFAULT_PARAMS: Bunch = Bunch()

    def __init__(self, seed=None, **kwargs):
        """Initialize external input with parameter overrides.

        Args:
            seed: Optional random seed for stateful inputs with randomness
            **kwargs: Parameter overrides for DEFAULT_PARAMS

        Raises:
            ValueError: If unknown parameters are provided
        """
        # Create instance parameters by copying defaults and updating with kwargs
        self.params = self.DEFAULT_PARAMS.copy() if self.DEFAULT_PARAMS else Bunch()
        for key, value in kwargs.items():
            if key not in self.DEFAULT_PARAMS:
                raise ValueError(
                    f"Unknown parameter '{key}' for {self.__class__.__name__}. "
                    f"Available parameters: {list(self.DEFAULT_PARAMS.keys())}"
                )
            self.params[key] = value

        # Store seed for stateful inputs that need randomness
        self.seed = seed

    @abstractmethod
    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """Prepare input for simulation.

        Called once before simulation starts. Handles setup logic like:
        - Resolving state indices (for state-dependent inputs)
        - Initializing internal state (for stateful inputs)
        - Precomputing static data

        Args:
            network: Network instance with graph, dynamics, initial_state
            dt: Integration timestep

        Returns:
            Tuple of (input_data, input_state):

            input_data: Bunch
                Static precomputed data (stored outside scan carry):
                - state_indices: Which states to monitor (for state-dependent)
                - dt: Timestep (for stateful updates)
                - Other static precomputed data

            input_state: Bunch
                Mutable internal state (stored in scan carry):
                - Empty Bunch() for stateless inputs
                - Internal variables for stateful inputs (e.g., adaptation level)
        """
        pass

    @abstractmethod
    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        input_data: Bunch,
        input_state: Bunch,
        params: Bunch,
    ) -> jnp.ndarray:
        """Compute external input at current time.

        Called at each timestep during simulation to calculate the input signal.

        Args:
            t: Current simulation time
            state: Current network state [n_states, n_nodes]
            input_data: Precomputed static data from prepare()
            input_state: Current mutable internal state
            params: Input parameters (supports broadcasting)

        Returns:
            Input array with flexible shape that will be auto-broadcast:
            - Scalar: Broadcast to [N_OUTPUT_DIMS, n_nodes]
            - [n_nodes]: Reshape to [N_OUTPUT_DIMS, n_nodes]
            - [N_OUTPUT_DIMS, n_nodes]: Use directly

            Parameters in params can be:
            - Scalars: Same value for all nodes
            - [n_nodes] arrays: Node-specific values (spatial patterns)
        """
        pass

    @abstractmethod
    def update_state(
        self,
        input_data: Bunch,
        input_state: Bunch,
        new_state: jnp.ndarray,
    ) -> Bunch:
        """Update input internal state after integration step.

        Called after each integration step. For stateless inputs, this is a no-op
        (just return input_state unchanged). For stateful inputs, this updates
        internal variables based on the new state.

        Args:
            input_data: Precomputed static data from prepare()
            input_state: Current mutable internal state
            new_state: New network state after integration [n_states, n_nodes]

        Returns:
            Updated input_state as Bunch
        """
        pass

    def __repr__(self) -> str:
        """String representation of external input."""
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"

    def _plot_prepare(self, n_nodes: int, dt: float) -> Tuple[Bunch, Bunch]:
        """Lightweight prepare for the plot path (no network required).

        Subclasses that need precomputed data (e.g. interpolators) override this.
        Default returns empty Bunches, which works for all stateless parametric
        inputs.
        """
        return Bunch(), Bunch()

    def plot(
        self,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        dt: float = None,
        n_nodes: int = 1,
        ax=None,
    ):
        """Plot the input signal over [t0, t1] for quick inspection.

        Calls ``compute`` directly with a zero state and empty input_data /
        input_state (or whatever ``_plot_prepare`` returns). State-dependent or
        stateful inputs will therefore show open-loop behavior only.

        Args:
            t0, t1: Time interval to plot.
            dt: Sampling step. Defaults to (t1 - t0) / 500.
            n_nodes: Number of nodes to render (lets spatial patterns show up).
            ax: Matplotlib axis (or array of axes for multi-dim outputs). Created
                if None.

        Returns:
            The matplotlib axis (or array of axes) used for plotting.
        """
        import matplotlib.pyplot as plt

        if dt is None:
            dt = (t1 - t0) / 500.0

        input_data, input_state = self._plot_prepare(n_nodes, dt)
        state = jnp.zeros((1, n_nodes))

        ts = jnp.arange(t0, t1, dt)
        signals = jnp.stack(
            [
                self.compute(float(t), state, input_data, input_state, self.params)
                for t in ts
            ]
        )  # [T, N_OUTPUT_DIMS, n_nodes]

        n_dims = signals.shape[1]
        if ax is None:
            fig, ax = plt.subplots(n_dims, 1, sharex=True, squeeze=False)
            ax = ax[:, 0]
        else:
            ax = ax if hasattr(ax, "__len__") else [ax]

        show_legend = n_nodes <= 8
        for d in range(n_dims):
            for i in range(n_nodes):
                ax[d].plot(ts, signals[:, d, i], label=f"node {i}" if show_legend else None)
            ax[d].set_ylabel(f"input[{d}]" if n_dims > 1 else "input")
            if show_legend and n_nodes > 1:
                ax[d].legend(loc="best", fontsize="small")
        ax[-1].set_xlabel("time")
        ax[0].set_title(repr(self))

        return ax if n_dims > 1 else ax[0]
