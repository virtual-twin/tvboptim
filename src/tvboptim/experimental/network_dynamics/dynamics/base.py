"""Abstract base class for neural dynamics models with multi-coupling support.

This version extends the original AbstractDynamics with support for multiple
named coupling inputs, enabling more flexible network architectures.
"""

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Tuple, Union

import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve

from ..core.bunch import Bunch

_REPARAMETERIZATION_DOCS = (
    "docs/basics/parameters_and_optimization.qmd ('Fitting a derived quantity')"
)


class AbstractDynamics(ABC):
    """Abstract base class for neural dynamics models with multi-coupling support.

    This base class extends the original AbstractDynamics with the ability to declare
    and receive multiple named coupling inputs via Bunch objects, enabling flexible
    network architectures with different coupling mechanisms.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        Names of integrated state variables
    INITIAL_STATE : tuple of float
        Default initial values for integrated states
    AUXILIARY_NAMES : tuple of str
        Names of auxiliary (non-integrated) variables
    DEFAULT_PARAMS : Bunch
        Default parameter values as Bunch object
    DERIVED_PARAMS : tuple of str
        Names of values computed from the parameters by ``derived_params``.
        They are never set, swept or optimized directly.
    COUPLING_INPUTS : dict
        Dictionary of expected coupling inputs ``{name: n_dims}``
    EXTERNAL_INPUTS : dict
        Dictionary of external input specifications ``{name: n_dims}``
    VARIABLES_OF_INTEREST : tuple
        Variables to record (empty tuple = record all state variables)
    """

    # Integrated state variables (required)
    STATE_NAMES: Tuple[str, ...] = ()
    INITIAL_STATE: Tuple[float, ...] = ()

    # Auxiliary variables (optional)
    AUXILIARY_NAMES: Tuple[str, ...] = ()

    # Parameters
    DEFAULT_PARAMS: Bunch = Bunch()
    DERIVED_PARAMS: Tuple[str, ...] = ()

    # NEW: Multi-coupling support
    COUPLING_INPUTS: dict = {}  # e.g., {'structural': 1, 'modulatory': 1}

    # External inputs (independent of network coupling)
    EXTERNAL_INPUTS: dict = {}  # e.g., {'stimulus': 1, 'perturbation': 2}

    # Recording configuration
    VARIABLES_OF_INTEREST: Tuple[Union[str, int], ...] = ()  # Empty = record all

    def __init__(self, **kwargs):
        """Initialize dynamics with optional parameter and configuration overrides.

        Parameters
        ----------
        **kwargs : dict
            Parameter overrides for DEFAULT_PARAMS or configuration options such as
            VARIABLES_OF_INTEREST or INITIAL_STATE
        """
        # Handle special configuration parameters
        if "VARIABLES_OF_INTEREST" in kwargs:
            self.VARIABLES_OF_INTEREST = kwargs.pop("VARIABLES_OF_INTEREST")
            # Validate variables of interest
            try:
                self.get_variables_of_interest_indices()
            except ValueError as e:
                raise ValueError(f"Invalid VARIABLES_OF_INTEREST: {e}")

        # Handle INITIAL_STATE override
        if "INITIAL_STATE" in kwargs:
            initial_state = kwargs.pop("INITIAL_STATE")
            # Convert to tuple if needed
            if isinstance(initial_state, (int, float)):
                initial_state = (initial_state,)
            elif not isinstance(initial_state, tuple):
                initial_state = tuple(initial_state)

            # Validate length matches STATE_NAMES
            if len(initial_state) != len(self.STATE_NAMES):
                raise ValueError(
                    f"INITIAL_STATE length ({len(initial_state)}) must match "
                    f"STATE_NAMES length ({len(self.STATE_NAMES)})"
                )
            self.INITIAL_STATE = initial_state

        # Create instance parameters by copying defaults and updating with kwargs
        self.params = self.DEFAULT_PARAMS.copy()
        for key, value in kwargs.items():
            if key in self.DERIVED_PARAMS:
                raise ValueError(
                    f"'{key}' is computed by {self.__class__.__name__}; set its "
                    f"primary parameters instead. Primary parameters: "
                    f"{list(self.DEFAULT_PARAMS.keys())}"
                )
            if key not in self.DEFAULT_PARAMS:
                raise ValueError(
                    f"Unknown parameter '{key}' for {self.__class__.__name__}. "
                    f"Available parameters: {list(self.DEFAULT_PARAMS.keys())}"
                )
            self.params[key] = value

    def __init_subclass__(cls, **kwargs):
        """Validate class attributes when subclassing."""
        super().__init_subclass__(**kwargs)

        # Skip validation for AbstractDynamics itself
        if cls.__name__ == "AbstractDynamics":
            return

        # Inherit attributes from parent if not explicitly defined
        if hasattr(cls, "__bases__") and len(cls.__bases__) > 0:
            parent = cls.__bases__[0]
            if issubclass(parent, AbstractDynamics) and parent != AbstractDynamics:
                # Inherit attributes that weren't redefined
                if not cls.STATE_NAMES:
                    cls.STATE_NAMES = parent.STATE_NAMES
                if not cls.INITIAL_STATE:
                    cls.INITIAL_STATE = parent.INITIAL_STATE
                if not cls.AUXILIARY_NAMES and hasattr(parent, "AUXILIARY_NAMES"):
                    cls.AUXILIARY_NAMES = parent.AUXILIARY_NAMES
                if not cls.COUPLING_INPUTS and hasattr(parent, "COUPLING_INPUTS"):
                    cls.COUPLING_INPUTS = parent.COUPLING_INPUTS
                if not cls.VARIABLES_OF_INTEREST and hasattr(
                    parent, "VARIABLES_OF_INTEREST"
                ):
                    cls.VARIABLES_OF_INTEREST = parent.VARIABLES_OF_INTEREST

                # Merge DEFAULT_PARAMS
                if hasattr(parent, "DEFAULT_PARAMS"):
                    merged_params = parent.DEFAULT_PARAMS.copy()
                    merged_params.update(cls.DEFAULT_PARAMS)
                    cls.DEFAULT_PARAMS = merged_params

        # Validate consistency of attributes
        if len(cls.STATE_NAMES) != len(cls.INITIAL_STATE):
            raise ValueError(f"{cls.__name__}: len(STATE_NAMES) != len(INITIAL_STATE)")

        # DERIVED_PARAMS uses ordinary attribute inheritance, unlike the
        # "empty means inherit" block above: an explicit () must be able to
        # replace a parent declaration when a value is promoted to primary.
        cls._validate_derived_declaration()

    @classmethod
    def _validate_derived_declaration(cls) -> None:
        """Check DERIVED_PARAMS format, disjointness and hook presence."""
        names = cls.DERIVED_PARAMS
        if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
            raise TypeError(
                f"{cls.__name__}.DERIVED_PARAMS must be a tuple of strings, "
                f"e.g. ('gain',); got {names!r}"
            )
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"{cls.__name__}.DERIVED_PARAMS has duplicate names: {duplicates}"
            )
        overlap = [n for n in names if n in cls.DEFAULT_PARAMS]
        if overlap:
            raise ValueError(
                f"{cls.__name__}: {overlap} appear in both DEFAULT_PARAMS and "
                f"DERIVED_PARAMS. A value is either primary or derived; to make "
                f"a derived value free, remove it from DERIVED_PARAMS and from "
                f"the derived_params() output."
            )
        if names and cls.derived_params is AbstractDynamics.derived_params:
            raise TypeError(
                f"{cls.__name__} declares DERIVED_PARAMS={names} but does not "
                f"implement or inherit derived_params()."
            )

    @property
    def all_variable_names(self) -> Tuple[str, ...]:
        """All variable names (integrated + auxiliary)."""
        return self.STATE_NAMES + self.AUXILIARY_NAMES

    @property
    def N_STATES(self) -> int:
        """Total number of variables (integrated + auxiliary)."""
        return len(self.STATE_NAMES)

    @property
    def N_AUXILIARIES(self) -> int:
        """Total number of variables (integrated + auxiliary)."""
        return len(self.AUXILIARY_NAMES)

    @property
    def n_variables(self) -> int:
        """Total number of variables (integrated + auxiliary)."""
        return self.N_STATES + self.N_AUXILIARIES

    @property
    def has_auxiliaries(self) -> bool:
        """Whether this dynamics has auxiliary variables."""
        return self.N_AUXILIARIES > 0

    def get_default_initial_state(self, n_nodes: int = 1) -> jnp.ndarray:
        """Get default initial state array.

        Args:
            n_nodes: Number of network nodes

        Returns:
            Initial state array of shape [N_STATES, n_nodes]
        """
        initial = jnp.array(self.INITIAL_STATE)
        if n_nodes == 1:
            return initial[:, None]
        else:
            return jnp.broadcast_to(initial[:, None], (self.N_STATES, n_nodes))

    def set_variables_of_interest(
        self, variables_of_interest: Tuple[Union[str, int], ...]
    ) -> None:
        """Update which variables are recorded in the trajectory output.

        Validates the new selection against STATE_NAMES + AUXILIARY_NAMES and
        assigns it. Any previously prepared model captured the old selection
        and must be rebuilt by calling ``prepare()`` again.

        Args:
            variables_of_interest: Tuple of variable names or indices to record.
                Empty tuple means "record all state variables" (default behavior).

        Raises:
            ValueError: If any entry is not a valid name or in-range index.
        """
        previous = self.VARIABLES_OF_INTEREST
        self.VARIABLES_OF_INTEREST = tuple(variables_of_interest)
        try:
            self.get_variables_of_interest_indices()
        except ValueError as e:
            self.VARIABLES_OF_INTEREST = previous
            raise ValueError(f"Invalid VARIABLES_OF_INTEREST: {e}")

    def get_variables_of_interest_indices(self) -> Tuple[int, ...]:
        """Get indices of variables to record.

        Returns:
            Tuple of indices into all_variable_names. If VARIABLES_OF_INTEREST is empty,
            returns only state variable indices (default behavior).
        """
        if not self.VARIABLES_OF_INTEREST:
            # Default: record only state variables (not auxiliaries)
            return tuple(range(self.N_STATES))

        indices = []
        for var in self.VARIABLES_OF_INTEREST:
            if isinstance(var, int):
                if 0 <= var < self.n_variables:
                    indices.append(var)
                else:
                    raise ValueError(
                        f"Variable index {var} out of range [0, {self.n_variables})"
                    )
            elif isinstance(var, str):
                if var in self.all_variable_names:
                    indices.append(self.all_variable_names.index(var))
                else:
                    raise ValueError(
                        f"Variable name '{var}' not found in {self.all_variable_names}"
                    )
            else:
                raise ValueError(
                    f"Variable of interest must be int or str, got {type(var)}"
                )

        return tuple(indices)

    def name_to_index(self, names: Union[str, Tuple[str, ...], list]) -> jnp.ndarray:
        """Convert state names to indices.

        Args:
            names: Single state name, tuple/list of state names, or empty list

        Returns:
            Array of indices corresponding to the state names

        Examples:
            For Lorenz with STATE_NAMES = ("x", "y", "z"):
            name_to_index("x") -> jnp.array([0])
            name_to_index(["y", "x"]) -> jnp.array([1, 0])
            name_to_index(("z", "y")) -> jnp.array([2, 1])
            name_to_index([]) -> jnp.array([], dtype=int)
        """
        if isinstance(names, str):
            # Single name
            if names not in self.STATE_NAMES:
                raise ValueError(
                    f"State name '{names}' not found in {self.STATE_NAMES}"
                )
            return jnp.array([self.STATE_NAMES.index(names)], dtype=int)

        elif isinstance(names, (list, tuple)):
            # Multiple names (including empty)
            if len(names) == 0:
                return jnp.array([], dtype=int)

            indices = []
            for name in names:
                if not isinstance(name, str):
                    raise ValueError(
                        f"All names must be strings, got {type(name)} for '{name}'"
                    )
                if name not in self.STATE_NAMES:
                    raise ValueError(
                        f"State name '{name}' not found in {self.STATE_NAMES}"
                    )
                indices.append(self.STATE_NAMES.index(name))
            return jnp.array(indices, dtype=int)

        else:
            raise ValueError(f"names must be str, list, or tuple, got {type(names)}")

    def derived_params(self, params: Bunch) -> Bunch:
        """Compute derived values from the live parameters.

        Override this to return a ``Bunch`` with exactly the names in
        ``DERIVED_PARAMS``. Only derived values are returned; the framework
        merges them with the primary parameters before calling ``dynamics()``.

        The hook runs inside jit/grad/vmap once per solve, so it must be pure
        JAX: read parameters from ``params`` (not ``self.params``) and do not
        mutate the input or the instance. Constants fixed at construction time
        may be read from ``self``.

        Parameters
        ----------
        params : Bunch
            Primary parameters, e.g. ``config.dynamics``.

        Returns
        -------
        Bunch
            Derived values keyed by the names in ``DERIVED_PARAMS``.
        """
        return Bunch()

    def resolve_params(self, params: Mapping, location: Optional[str] = None) -> Bunch:
        """Return primary and derived parameters as one ``Bunch``.

        This is what ``dynamics()`` receives during a solve. Use it to inspect
        derived values or to call ``dynamics()`` directly. Not meant to be
        overridden; implement ``derived_params`` instead.

        Parameters
        ----------
        params : Bunch
            Primary parameters, e.g. ``model.params`` or ``config.dynamics``.
            Must not contain derived names.
        location : str, optional
            Where ``params`` came from (e.g. ``"config.dynamics"``), used in
            error messages.

        Returns
        -------
        Bunch
            A new ``Bunch``; ``params`` is left untouched. For models without
            derived parameters, ``params`` is returned unchanged.

        Examples
        --------
        >>> resolved = model.resolve_params(config.dynamics)
        >>> model.dynamics(t, state, resolved, coupling, external)
        """
        if type(self).derived_params is AbstractDynamics.derived_params:
            return params

        name = self.__class__.__name__
        where = f" in {location}" if location else ""
        derived_names = self.DERIVED_PARAMS

        given = [n for n in derived_names if n in params]
        if given:
            raise ValueError(
                f"{given}{where} are computed by {name}.derived_params() and "
                f"cannot be set, swept or optimized directly. Set or fit the "
                f"primary parameters instead. To fit a derived quantity, "
                f"reparameterize the model; see {_REPARAMETERIZATION_DOCS}."
            )

        derived = self.derived_params(params)
        if not isinstance(derived, Bunch):
            raise TypeError(
                f"{name}.derived_params() must return a Bunch, "
                f"got {type(derived).__name__}"
            )
        missing = [n for n in derived_names if n not in derived]
        extra = [n for n in derived if n not in derived_names]
        if missing or extra:
            raise ValueError(
                f"{name}.derived_params() must return exactly DERIVED_PARAMS="
                f"{derived_names}; missing {missing}, unexpected {extra}"
            )

        resolved = Bunch(params)
        resolved.update(derived)
        return resolved

    @abstractmethod
    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Compute state derivatives and auxiliary variables.

        Parameters
        ----------
        t : float
            Current time
        state : jnp.ndarray
            Current state with shape ``[N_STATES, n_nodes]``
        params : Bunch
            Model parameters as Bunch object (supports broadcasting). Includes
            derived values, see ``resolve_params``.
        coupling : Bunch
            Named coupling inputs accessed as attributes (e.g., ``coupling.structural``,
            ``coupling.modulatory``). Missing couplings are automatically filled with zeros.
        external : Bunch
            Named external inputs accessed as attributes (e.g., ``external.stimulus``,
            ``external.perturbation``). Missing inputs are automatically filled with zeros.

        Returns
        -------
        derivatives : jnp.ndarray or tuple
            If ``N_AUXILIARIES == 0``: Returns derivatives array with shape ``[N_STATES, n_nodes]``

            If ``N_AUXILIARIES > 0``: Returns tuple ``(derivatives, auxiliaries)`` where:

            - derivatives has shape ``[N_STATES, n_nodes]``
            - auxiliaries has shape ``[N_AUXILIARIES, n_nodes]``

        Notes
        -----
        - If COUPLING_INPUTS is empty, the model has no coupling
        - Coupling arrays have shape ``[n_dims, n_nodes]`` where n_dims is declared in COUPLING_INPUTS
        - If EXTERNAL_INPUTS is empty, the model has no external inputs
        - External input arrays have shape ``[n_dims, n_nodes]`` where n_dims is declared in EXTERNAL_INPUTS
        """
        pass

    def verify(self, n_nodes: int = 1, verbose: bool = True) -> bool:
        """Verify that dynamics implementation is correct.

        Tests the dynamics function with default initial state and parameters
        at t=0 with zero coupling to ensure it runs without errors and returns
        the expected output format.

        Args:
            n_nodes: Number of nodes to test with
            verbose: Whether to print verification details

        Returns:
            True if verification passes, False otherwise
        """
        try:
            # Get test inputs
            state = self.get_default_initial_state(n_nodes)
            params = self.resolve_params(self.params, "model.params")

            # Create coupling Bunch with zeros for all declared inputs
            coupling = Bunch()
            for name, n_dims in self.COUPLING_INPUTS.items():
                coupling[name] = jnp.zeros((n_dims, n_nodes))

            # Create external Bunch with zeros for all declared inputs
            external = Bunch()
            for name, n_dims in self.EXTERNAL_INPUTS.items():
                external[name] = jnp.zeros((n_dims, n_nodes))

            if verbose:
                print(f"Verifying {self.__class__.__name__}:")
                print(f"  State shape: {state.shape}")
                print(f"  Coupling inputs: {list(self.COUPLING_INPUTS.keys())}")
                print(f"  External inputs: {list(self.EXTERNAL_INPUTS.keys())}")
                print(f"  Expected auxiliaries: {self.N_AUXILIARIES}")

            # Call dynamics
            result = self.dynamics(0.0, state, params, coupling, external)

            # Check return format
            if self.N_AUXILIARIES == 0:
                # Should return just derivatives
                if isinstance(result, tuple):
                    if verbose:
                        print("  ❌ ERROR: Expected single array but got tuple")
                    return False
                derivatives = result
                if derivatives.shape != (self.N_STATES, n_nodes):
                    if verbose:
                        print(
                            f"  ❌ ERROR: Expected derivatives shape {(self.N_STATES, n_nodes)}, got {derivatives.shape}"
                        )
                    return False
            else:
                # Should return (derivatives, auxiliaries)
                if not isinstance(result, tuple) or len(result) != 2:
                    if verbose:
                        print(
                            "  ❌ ERROR: Expected tuple of (derivatives, auxiliaries)"
                        )
                    return False
                derivatives, auxiliaries = result
                if derivatives.shape != (self.N_STATES, n_nodes):
                    if verbose:
                        print(
                            f"  ❌ ERROR: Expected derivatives shape {(self.N_STATES, n_nodes)}, got {derivatives.shape}"
                        )
                    return False
                if auxiliaries.shape != (self.N_AUXILIARIES, n_nodes):
                    if verbose:
                        print(
                            f"  ❌ ERROR: Expected auxiliaries shape {(self.N_AUXILIARIES, n_nodes)}, got {auxiliaries.shape}"
                        )
                    return False

            if verbose:
                print("  ✅ Verification passed!")
            return True

        except Exception as e:
            if verbose:
                print(f"  ❌ ERROR: {e}")
            return False

    def simulate(
        self,
        t0: float = 0.0,
        t1: float = 10.0,
        dt: float = 0.01,
        n_nodes: int = 1,
        params: Optional[Bunch] = None,
        initial_state: Optional[jnp.ndarray] = None,
        solver=None,
        **solver_kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate dynamics using Diffrax.

        Args:
            t0: Start time
            t1: End time
            dt: Time step for output
            n_nodes: Number of nodes to simulate
            params: Primary parameters (default: self.params)
            initial_state: Initial state (default: from get_default_initial_state)
            solver: Diffrax solver (default: Dopri5)
            **solver_kwargs: Additional solver arguments

        Returns:
            times: [n_timesteps] - Time points
            trajectory: [n_timesteps, n_states, n_nodes] - State trajectory
        """
        if params is None:
            params = self.params
        params = self.resolve_params(params)

        if initial_state is None:
            initial_state = self.get_default_initial_state(n_nodes)

        if solver is None:
            solver = Dopri5()

        # Create zero coupling Bunch for all declared coupling inputs
        coupling_zeros = Bunch()
        for name, n_dims in self.COUPLING_INPUTS.items():
            coupling_zeros[name] = jnp.zeros((n_dims, n_nodes))

        # Create zero external Bunch for all declared external inputs
        external_zeros = Bunch()
        for name, n_dims in self.EXTERNAL_INPUTS.items():
            external_zeros[name] = jnp.zeros((n_dims, n_nodes))

        def ode_func(t, y, args):
            """ODE function for Diffrax - returns only derivatives."""
            result = self.dynamics(t, y, params, coupling_zeros, external_zeros)
            # Extract only derivatives if auxiliaries are returned
            if isinstance(result, tuple):
                return result[0]  # derivatives only
            return result

        # Set up time points - ensure we don't exceed t1
        times = jnp.arange(
            t0, t1 + dt / 2, dt
        )  # Add dt/2 to handle floating point precision
        save_at = SaveAt(ts=times)

        # Solve ODE
        term = ODETerm(ode_func)
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=initial_state,
            saveat=save_at,
            **solver_kwargs,
        )

        return times, solution.ys  # shape: [n_timesteps, n_states, n_nodes]

    def plot(
        self,
        t0: float = 0.0,
        t1: float = 10.0,
        dt: float = 0.01,
        n_nodes: int = 1,
        figsize: Tuple[int, int] = (10, 6),
        **simulate_kwargs,
    ) -> None:
        """Plot dynamics timeseries for state variables.

        Note: Currently only plots state variables (not auxiliaries) as auxiliaries
        are not computed during simulation.

        Args:
            t0: Start time
            t1: End time
            dt: Time step
            n_nodes: Number of nodes to simulate
            figsize: Figure size (width, height)
            **simulate_kwargs: Additional arguments for simulate()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Simulate dynamics
        times, trajectory = self.simulate(
            t0=t0, t1=t1, dt=dt, n_nodes=n_nodes, **simulate_kwargs
        )

        # Plot only state variables (trajectory only contains states)
        # Create plot
        fig, axes = plt.subplots(self.N_STATES, 1, figsize=figsize, sharex=True)
        if self.N_STATES == 1:
            axes = [axes]  # Make iterable for single plot

        for state_idx, state_name in enumerate(self.STATE_NAMES):
            ax = axes[state_idx]

            # Plot each node
            for node in range(n_nodes):
                label = f"Node {node}" if n_nodes > 1 else None
                ax.plot(times, trajectory[:, state_idx, node], label=label)

            ax.set_ylabel(state_name)
            ax.grid(True, alpha=0.3)

            if n_nodes > 1:
                ax.legend()

        axes[-1].set_xlabel("Time")
        plt.suptitle(f"{self.__class__.__name__} Dynamics")
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        """String representation of dynamics."""
        coupling_str = (
            f", coupling_inputs={list(self.COUPLING_INPUTS.keys())}"
            if self.COUPLING_INPUTS
            else ""
        )
        external_str = (
            f", external_inputs={list(self.EXTERNAL_INPUTS.keys())}"
            if self.EXTERNAL_INPUTS
            else ""
        )
        return (
            f"{self.__class__.__name__}("
            f"states={self.N_STATES}, "
            f"auxiliaries={self.N_AUXILIARIES}"
            f"{coupling_str}"
            f"{external_str})"
        )
