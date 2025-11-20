"""Diffrax-based solvers with advanced features."""

import diffrax
from diffrax import SaveAt

from .base import AbstractSolver


class DiffraxSolver(AbstractSolver):
    """Wrapper for Diffrax solvers with advanced features.

    This class forwards the most commonly used diffeqsolve parameters explicitly
    and allows access to all other parameters via **kwargs.

    Important Notes:
        - When max_steps is specified, Diffrax may pad solution arrays (ts, ys)
          with inf values beyond the actual integration steps taken.
        - Users should filter finite values in post-processing if needed:

          >>> finite_mask = jnp.isfinite(solution.ts)
          >>> ts_filtered = solution.ts[finite_mask]
          >>> ys_filtered = solution.ys[finite_mask]

        - To avoid inf-padding, explicitly specify saveat with exact time points:

          >>> solver = DiffraxSolver(
          ...     solver=diffrax.Euler(),
          ...     saveat=diffrax.SaveAt(ts=jnp.linspace(0, 100, 1000))
          ... )
    """
    
    def __init__(self, solver, 
                 # Common parameters made explicit
                 saveat=None, 
                 stepsize_controller=None, 
                 max_steps=4096,
                 # Forward everything else
                 **kwargs):
        """Initialize with Diffrax solver instance and options.
        
        Args:
            solver: Diffrax solver instance (e.g., diffrax.Heun())
            saveat: SaveAt instance for output control
            stepsize_controller: Step size controller for adaptive stepping
            max_steps: Maximum integration steps
            **kwargs: Additional arguments passed to diffeqsolve (e.g., adjoint, event, 
                     throw, progress_meter, solver_state, controller_state, made_jump, etc.)
                     See https://docs.kidger.site/diffrax/api/diffeqsolve/ for full list.
        """
        self.solver = solver
        self.saveat = saveat if saveat is not None else SaveAt(steps=True)
        self.stepsize_controller = stepsize_controller if stepsize_controller is not None else diffrax.ConstantStepSize()
        self.max_steps = max_steps
        
        # Store additional kwargs for diffeqsolve
        self.diffrax_kwargs = kwargs