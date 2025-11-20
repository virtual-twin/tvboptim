import numpy as np
import jax
import jax.numpy as jnp
import optax

from tvboptim.types.stateutils import partition_state, combine_state
from tvboptim.utils import format_pytree_as_string
# from tvb_fit.base.parameter import Parameters

import copy
from functools import partial
# from tvboptim.optim.callbacks import * 

class OptaxOptimizer():
    """
    JAX-based parameter optimization using Optax optimizers with automatic differentiation.
    
    OptaxOptimizer provides a high-level interface for optimizing model parameters
    using any Optax optimizer (Adam, SGD, RMSprop, etc.). It automatically handles
    parameter partitioning, gradient computation, and state management while
    supporting both forward-mode and reverse-mode automatic differentiation.
    
    Parameters
    ----------
    loss : callable
        Loss function to minimize. Should accept a state parameter and return
        a scalar loss value. Signature: loss(state) -> scalar or (scalar, aux_data)
        if has_aux=True.
    optimizer : optax.GradientTransformation
        Optax optimizer instance (e.g., optax.adam(0.001), optax.sgd(0.01)).
        Defines the optimization algorithm and hyperparameters.
    callback : callable, optional
        Optional callback function called after each optimization step.
        Signature: callback(step, diff_state, static_state, fitting_data, aux_data, 
        loss_value, grads) -> (stop_flag, new_diff_state, new_static_state).
        Default is None, see the callbacks module for many useful callbacks.
    has_aux : bool, optional
        Whether the loss function returns auxiliary data along with the loss value.
        If True, loss should return (loss_value, aux_data). Default is False.
    
    
    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import optax
    >>> from tvboptim.types.parameter import Parameter
    >>> 
    >>> # Define loss function
    >>> def mse_loss(state):
    ...     prediction = state['weight'] * state['input'] + state['bias']
    ...     target = 2.5
    ...     return jnp.mean((prediction - target) ** 2)
    >>> 
    >>> # Define parameter state
    >>> state = {
    ...     'weight': Parameter("weight", 1.0, free=True),
    ...     'bias': Parameter("bias", 0.0, free=True),
    ...     'input': 1.5  # Static parameter
    ... }
    >>> 
    >>> # Create optimizer
    >>> opt = OptaxOptimizer(
    ...     loss=mse_loss,
    ...     optimizer=optax.adam(learning_rate=0.01)
    ... )
    >>> 
    >>> # Run optimization
    >>> final_state, history = opt.run(state, max_steps=1000)
    >>> print(f"Optimized weight: {final_state['weight']}")
    >>> 
    >>> # With auxiliary data and callback
    >>> def loss_with_aux(state):
    ...     pred = state['weight'] * state['input'] + state['bias']
    ...     loss = jnp.mean((pred - 2.5) ** 2)
    ...     aux = {'prediction': pred, 'error': pred - 2.5}
    ...     return loss, aux
    >>> 
    >>> def monitor_callback(step, diff_state, static_state, fitting_data, 
    ...                      aux_data, loss_value, grads):
    ...     if step % 100 == 0:
    ...         print(f"Step {step}: Loss = {loss_value}")
    ...     # Early stopping condition
    ...     stop = loss_value < 1e-6
    ...     return stop, diff_state, static_state
    >>> 
    >>> opt_aux = OptaxOptimizer(
    ...     loss=loss_with_aux,
    ...     optimizer=optax.adam(0.01),
    ...     callback=monitor_callback,
    ...     has_aux=True
    ... )
    >>> final_state, history = opt_aux.run(state, max_steps=1000, mode="rev")
    
    Notes
    -----
    **Parameter Partitioning:**
    
    The optimizer automatically partitions the state into:
    - **diff_state**: Parameters marked as free=True (optimized)
    - **static_state**: Parameters marked as free=False (constant)
    
    Only free parameters are optimized, while static parameters remain unchanged
    throughout the optimization process.
    
    **Differentiation Modes:**
    
    - **"rev"** (default): Reverse-mode AD, efficient for many parameters
    - **"fwd"**: Forward-mode AD, efficient for few parameters or when gradients 
      are needed w.r.t. many outputs
    """
    def __init__(self, loss, optimizer, callback=None, has_aux=False):
        self.loss = loss
        self.optimizer = optimizer
        self.callback = callback
        self.has_aux = has_aux

    def run(self, state, max_steps=1, mode="rev"):
        """
        Execute parameter optimization for the specified number of steps.
        
        Performs gradient-based optimization of free parameters in the state
        using the configured Optax optimizer. Automatically handles parameter
        partitioning, gradient computation, and state updates.
        
        Parameters
        ----------
        state : PyTree
            Initial parameter state containing both free and static parameters.
            Free parameters (marked with free=True) will be optimized.
        max_steps : int, optional
            Maximum number of optimization steps to perform. Default is 1.
        mode : {"rev", "fwd"}, optional
            Automatic differentiation mode. "rev" for reverse-mode (default),
            "fwd" for forward-mode. Default is "rev".
        
        Returns
        -------
        tuple
            A tuple containing:
            
            - **final_state** (PyTree): Optimized parameter state with
              updated free parameters and unchanged static parameters.
            - **fitting_data** (dict): Dictionary containing optimization history
              and metadata collected during the optimization process.
        
        Notes
        -----
        **Gradient Computation:**
        
        The method automatically selects appropriate gradient computation based
        on the mode parameter and loss function characteristics. Reverse-mode
        is typically preferred for parameter optimization scenarios.
        """
        diff_state, static_state = partition_state(state)

        def __loss(diff_state, static_state):
            state = combine_state(diff_state, static_state)
            return self.loss(state)
            # return (self.loss(state), (None, None))
            
        # _loss = jax.jit(__loss, static_argnums=(1,))
        _loss = jax.jit(__loss)
        
        if mode == "rev":
            def value_and_grad(loss, argnums=0, has_aux=False):
                vgf = jax.value_and_grad(loss, argnums=argnums, has_aux=has_aux)
                if has_aux:
                     return vgf
                else:
                    def _fun(*args, **kwargs):
                        val, grads = vgf(*args, **kwargs)
                        return (val, None), grads
                    return _fun

        elif mode == "fwd":
            def value_and_grad(loss, argnums=0, has_aux=False):
                return value_and_grad_fwd(loss, argnums=argnums, has_aux=has_aux)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        v_g_fun = value_and_grad(_loss, argnums=0, has_aux=self.has_aux)
        # v_g_fun = jax.jit(value_and_grad(_loss, argnums=0, has_aux=self.has_aux), static_argnums=(1,))

        def step(diff_state, static_state, opt_state):
            # out = value_and_grad(_loss, argnums=0, has_aux=self.has_aux)(diff_state, static_state)
            out = v_g_fun(diff_state, static_state)
            (loss_value, aux_data), grads = out
            # if self.has_aux:
            #     (loss_value, aux_data), grads = out
            # else:
            #     loss_value, grads = out
            #     aux_data = None
            def f(p):
                return _loss(diff_state, static_state)
            
            updates, opt_state = self.optimizer.update(grads, opt_state, diff_state, value=loss_value, grad=grads, value_fn=f)

            diff_state = optax.apply_updates(diff_state, updates)
            return diff_state, grads, opt_state, loss_value, aux_data

        opt_state = self.optimizer.init(diff_state)

        fitting_data = dict()  # store data during fitting
        for i in range(max_steps):
            diff_state, grads, opt_state, loss_value, aux_data = step(diff_state, static_state, opt_state)
            # print(format_pytree_as_string(diff_state, hide_none=True, name="FreeState", show_array_values=True))
            # print(format_pytree_as_string(diff_state, hide_none=True, name="FreeState", show_array_values=True))
            # fitting_data.append(diff_state)
            if self.callback is not None:
                            stop, diff_state, static_state = self.callback(i, diff_state, static_state, fitting_data, aux_data, loss_value, grads)
                            if stop:
                                print("Stopping due to callback")
                                break
        return combine_state(diff_state, static_state), fitting_data

def value_and_grad_fwd(fun, argnums=0, has_aux=False):
    grad_fun = jax.jacfwd(fun, argnums=argnums, has_aux=has_aux)
    if has_aux:
        def _fun(*args, **kwargs):
            val, _ = fun(*args, **kwargs)
            grads, aux = grad_fun(*args, **kwargs)
            return (val, aux), grads
    else:
        def _fun(*args, **kwargs):
            val = fun(*args, **kwargs)
            grads = grad_fun(*args, **kwargs)
            return (val, None), grads
    return _fun

# class Optimizer():

#     def __init__(self, 
#                  gm, 
#                  observation = None, 
#                  metric = None,
#                  callback = None,
#                  optimizer = None
#                  ):
#         self.gm = gm
#         self.observation = observation
#         self.metric = metric
#         self.callback = callback
#         self.optimizer = optimizer
    
#     def __call__(self, *args, **kwargs):
#         return self.fit(*args, **kwargs)
    
#     def validate(self):
#         pred, params = self.gm.run()
#         assert isinstance(self.metric(self.observation, pred, params, params), float), f"Metric must return a scalar float"        
#         return 
    
#     def fit(self):
#         pass

# class OptaxOptimizer(Optimizer):

#     def simulate(self,parameters, ics, metadata):
#         result, ics_new = self.gm.kernel(self.gm.preprocess(parameters), ics, noise = self.gm.noise)
#         prediction = self.gm.observation_model.operation(result, parameters, metadata = metadata)
#         return prediction, ics_new

#     def loss(self, parameters, ics, metadata):
#         prediction, ics_new = self.simulate(parameters, ics, metadata)
#         loss_value = self.metric(prediction, self.observation, parameters, self.gm.params)
#         return (loss_value, (prediction, ics_new))

#     def fit(self, iter_max = 1, mode = "rev", initial_parameters = None, initial_conditions = None, update_noise = False):
#         """
#         Fit the prediction of the generative model to the observation.

#         **Arguments:**

#         * *iter_max*: Maximum number of iterations
#         * *mode*: 'rev' - reverse mode, 'fwd' - forward mode, 'lin' - linear mode
#         * *initial_parameters*: Initial parameters for the optimization if None then the parameters are taken from the generative model.
#         * *initial_conditions*: Initial conditions for the optimization if None then the initial conditions are taken from the generative model.  
#         """
#         @jax.jit
#         def simulate(parameters, ics, noise, metadata):
#             result, ics_new = self.gm.kernel(self.gm.preprocess(parameters), ics, noise = noise)
#             prediction = self.gm.observation_model.operation(result, parameters, metadata = metadata)
#             return prediction, ics_new

#         @jax.jit
#         def loss(parameters, ics, noise, metadata):
#             prediction, ics_new = simulate(parameters, ics, noise, metadata)
#             loss_value = self.metric(prediction, self.observation, parameters, self.gm.params)
#             return (loss_value, (prediction, ics_new))

#         if mode == "rev":
#             def value_and_grad(loss, params, argnums=0, has_aux=True):
#                 return jax.value_and_grad(loss, argnums=argnums, has_aux=has_aux)
#         elif mode == "rev_state":
#             def value_and_grad(loss, params, argnums=0, has_aux=True):
#                 return optax.value_and_grad_from_state(loss, argnums=argnums, has_aux=has_aux)
#         elif mode == "fwd":
#             def value_and_grad(loss, params, argnums=0, has_aux=True):
#                 return value_and_grad_fwd(loss, params, argnums=argnums, has_aux=has_aux)
#         else:
#             raise NotImplementedError(f"Mode {mode} not implemented. Must be 'rev', 'fwd', or 'lin'")

#         def step(parameters, ics, noise, metadata, opt_state):
#             out = value_and_grad(loss, parameters, argnums=0, has_aux=True)(parameters, ics, noise, metadata)
#             (loss_value, (prediction, ics_new)), grads = out
#             def f(p):
#                 return loss(p, ics, noise, metadata)[0]
#             updates, opt_state = self.optimizer.update(grads, opt_state, parameters, value=loss_value, grad=grads, value_fn=f)
#             # updates, opt_state = self.optimizer.update(grads, opt_state, parameters)
#             parameters = optax.apply_updates(parameters, updates)
#             return parameters, ics_new, opt_state, loss_value, prediction, grads
        
#         if initial_parameters is None:
#             parameters = self.gm.params
#         else:
#             parameters = initial_parameters

#         if initial_conditions is None:
#             ics = self.gm.initial_conditions
#         else:
#             ics = initial_conditions

#         opt_state = self.optimizer.init(parameters)
#         metadata = self.gm.observation_model.metadata
#         noise = self.gm.noise
#         fitting_data = dict() # a place to store data during fitting
#         for i in range(iter_max):
#             # ics need to be converted to tuple, otherwise recompilation will be triggered each iteration - alternative: unpack named tuple in state, history.
#             parameters, ics_new, opt_state, loss_value, prediction, grads = step(parameters, tuple(ics), noise, metadata, opt_state)
#             if update_noise:
#                 if hasattr(parameters, "nsig"):
#                     # print(f"Updating noise nsig = {parameters.nsig.value}")
#                     noise = self.gm.noise_generator(nsig = parameters.nsig.value)
#                 else:
#                     # print("Updating noise")
#                     noise = self.gm.noise_generator()
#             # Ensure ICs stay finite
#             if np.isfinite(ics_new[0]).all() and np.isfinite(ics_new[1]).all():
#                 ics = ics_new
#             if self.callback is not None:
#                 stop, parameters, ics, metadata = self.callback(i, parameters, ics, metadata, fitting_data, self.gm, prediction, loss_value, grads)
#                 if stop:
#                     print("Stopping due to callback")
#                     break
            
#         return parameters, ics, metadata, fitting_data


# def value_and_grad_fwd(fun, params, argnums=0, has_aux=False):
#     grad_fun = jax.jacfwd(fun)
#     def _fun(*args, **kwargs):
#         val = fun(params)
#         grads = grad_fun(*args, **kwargs)
#         return val, grads
#     return _fun

