import time

import numpy as np
import pandas as pd

from tvboptim.utils import format_pytree_as_string

# stop, parameters, ics, metadata = self.callback(i, parameters, ics, metadata, fitting_data, self.gm, prediction, loss_value, grads)


class AbstractCallback:
    def __init__(self, every=1) -> None:
        self.every = every

    def __call__(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        if i % self.every == 0:
            return self.do(
                i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
            )
        return False, diff_state, static_state

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        return False, diff_state, static_state


# class UpdateICsAndMetadataCallback(AbstractCallback):
#     def __init__(self, updater, every=1) -> None:
#         super().__init__(every)
#         self.updater = updater
#     def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
#         metadata_new, ics_new = self.updater(gm, parameters, ics)
#         return False, parameters, ics_new, metadata_new


class SavingCallback(AbstractCallback):
    def __init__(self, every=1, key="", save_fun=lambda *args: None) -> None:
        super().__init__(every)
        self.key = key
        self.save_fun = save_fun

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        if self.key not in fitting_data:
            fitting_data[self.key] = pd.DataFrame(columns=["step", "save"])

        fitting_data[self.key].loc[len(fitting_data[self.key])] = [
            i,
            self.save_fun(
                i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
            ),
        ]
        return False, diff_state, static_state


class SavingLossCallback(SavingCallback):
    def __init__(self, every=1, *args: None) -> None:
        super().__init__(
            every,
            key="loss",
            save_fun=lambda i,
            diff_state,
            static_state,
            fitting_data,
            aux_data,
            loss_value,
            grads: loss_value,
        )


class SavingParametersCallback(SavingCallback):
    def __init__(self, every=1, *args: None) -> None:
        super().__init__(
            every,
            key="parameters",
            save_fun=lambda i,
            diff_state,
            static_state,
            fitting_data,
            aux_data,
            loss_value,
            grads: diff_state,
        )


class TimingCallback(AbstractCallback):
    def __init__(self, every=1, key="timing", *args: None) -> None:
        super().__init__(every)
        self.key = key
        self.t0 = None

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        if self.key not in fitting_data:
            fitting_data[self.key] = pd.DataFrame(columns=["step", "save"])
        if self.t0 is None or i == 0:
            self.t0 = time.time()
        fitting_data[self.key].loc[len(fitting_data[self.key])] = [
            i,
            time.time() - self.t0,
        ]
        return False, diff_state, static_state


class StopTimeCallback(AbstractCallback):
    def __init__(self, every=1, time_limit=0) -> None:
        super().__init__(every)
        self.time_limit = time_limit
        self.t0 = None

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        if self.t0 is None or i == 0:
            self.t0 = time.time()
        if (time.time() - self.t0) > self.time_limit:
            print(
                f"Stopped at step {i} after reaching time limit of {self.time_limit} seconds"
            )
            return True, diff_state, static_state
        return False, diff_state, static_state


class SaveBestSeenCallback(AbstractCallback):
    def __init__(self, every=1, key="best", minimization=True) -> None:
        super().__init__(every)
        self.key = key
        self.minimization = minimization

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        _loss_value = loss_value
        if not self.minimization:
            _loss_value *= -1
        if self.key not in fitting_data:
            fitting_data[self.key] = (loss_value, i, diff_state, static_state)
        elif _loss_value < fitting_data[self.key][0]:
            fitting_data[self.key] = (loss_value, i, diff_state, static_state)
        return False, diff_state, static_state


class DefaultPrintCallback(AbstractCallback):
    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        print(f"Step {i}: {loss_value:6f}")
        return False, diff_state, static_state


class PrintParameterCallback(AbstractCallback):
    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        print(f"Step {i}, Loss={loss_value:6f} State:")
        print(
            format_pytree_as_string(
                diff_state, hide_none=True, name="Parameters", show_array_values=True
            )
        )
        return False, diff_state, static_state


# Todo -> upstream to format_pytree_as_string
# class PrintGlobalParametersCallback(AbstractCallback):
#     def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
#         print(f"Global parameters at Step {i}: {loss_value:6f}")
#         for p in parameters:
#             if np.prod(p.shape) == 1:
#                 print(f"{p}")
#             else:
#                 print(f"Norm {p.name}: {np.linalg.norm(p.value)}")
#         return False, diff_state, static_state


class PrintGradsCallback(AbstractCallback):
    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        print(f"Grads at Step {i}: {loss_value:6f}")
        print(
            format_pytree_as_string(
                grads, hide_none=True, name="GradientState", show_array_values=True
            )
        )

        return False, diff_state, static_state


class PrintGlobalGradsCallback(AbstractCallback):
    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        print(f"Global grads at Step {i}: {loss_value:6f}")
        for g in grads:
            if np.prod(g.shape) == 1:
                print(f"{g}")
            else:
                print(f"Norm {g.name}: {np.linalg.norm(g.value)}")
        return False, diff_state, static_state


class StopLossCallback(AbstractCallback):
    def __init__(self, every=1, stop_loss=0) -> None:
        super().__init__(every)
        self.stop_loss = stop_loss

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        if loss_value < self.stop_loss:
            print(f"Stopped at step {i} with loss {loss_value:6f}")
            return True, diff_state, static_state
        return False, diff_state, static_state


class StopConvergenceCallback(AbstractCallback):
    """
    Stop fitting if no improvement was seen for `patience` number of iterations. Improvement is defined by loss_new < loss_best - `min_delta`.
    """

    def __init__(self, every=1, patience=10, min_delta=10e-4) -> None:
        super().__init__(every)
        self.patience = patience
        self.min_delta = min_delta
        self.patience_count = 0
        self.best_loss = np.Inf

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        if i == 0:  # Reset on first iteration
            self.patience_count = 0
            self.best_loss = np.Inf

        if loss_value < self.best_loss - self.min_delta:
            self.best_loss = loss_value
            self.patience_count = 0
        else:
            self.patience_count += 1

        converged = self.patience_count > self.patience
        if converged:
            print(
                f"Stopped at step {i} with loss {loss_value:6f} due to no improvement after {self.patience} steps"
            )

        return converged, diff_state, static_state


# class PlotObservationCallback(AbstractCallback):
#     def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
#         plt.figure()
#         plt.plot(prediction[0].T, color = 'k', alpha = 0.25)
#         # plt.imshow(prediction[0].T)
#         plt.show()

#         return False, diff_state, static_state

# class PlotTSCallback(AbstractCallback):
#     """
#     Plots the time series of the simulator without applying the observation model
#     """
#     def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
#         # could be forwarded from inference to save double computing
#         res = gm.kernel(gm.preprocess(parameters), ics)[0]
#         has_multi_monitors = not hasattr(res, "time")
#         if has_multi_monitors:
#             n_mon = len(res)
#             n_svar = res[0].trace.shape[1]
#         else:
#             n_mon = 1
#             n_svar = res.trace.shape[1]

#         _, axs = plt.subplots(n_svar, n_mon)
#         try:
#             axs_flat = axs.flatten()
#         except:
#             axs_flat = [axs]
#         for m in range(n_mon):
#             for s in range(n_svar):
#                 ax = axs_flat[m+(s*(n_mon))]
#                 if has_multi_monitors:
#                     ax.plot(res[m].time, res[m].trace[:, s, :, 0], alpha = 0.1)
#                 else:
#                     ax.plot(res.time, res.trace[:, s, :, 0], alpha = 0.1)
#                 if s == 0:
#                     ax.set_title(f'Monitor {m}')
#                 if m == 0:
#                     ax.set_ylabel(f'state variable {s}')
#                 if s == n_svar-1:
#                     ax.set_xlabel('time')

#         plt.tight_layout()
#         plt.show()
#         return False, diff_state, static_state

# class PlotTSDiffCallback(AbstractCallback):
#     def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
#         res = gm(parameters)
#         _, ax = plt.subplots(1, 3, figsize=(20,4))
#         ax[0].plot(jnp.squeeze(target))
#         ax[0].set_title(f'Target')
#         ax[1].plot(jnp.squeeze(res))
#         ax[1].set_title(f'Prediction at step {i} with loss {loss_value:6f}')
#         ax[1].sharey(ax[0])
#         ax[2].plot(jnp.squeeze(target - res))
#         ax[2].set_title(f'Difference')
#         plt.tight_layout()
#         plt.show()
#         return False, diff_state, static_state


class MultiCallback(AbstractCallback):
    def __init__(self, callbacks, every=1) -> None:
        self.callbacks = callbacks
        self.every = every

    def do(
        self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
    ):
        for callback in self.callbacks:
            test, diff_state, static_state = callback(
                i, diff_state, static_state, fitting_data, aux_data, loss_value, grads
            )
            if test:
                return True, diff_state, static_state
        return False, diff_state, static_state
