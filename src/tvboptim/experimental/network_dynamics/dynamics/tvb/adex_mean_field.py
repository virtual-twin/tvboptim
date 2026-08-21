"""
Mean-Field of interconnected excitatory and inhibitory populations of Adaptive-Exponential neurons.

References:
    - di Volo et al., 2019. Biologically realistic mean-field models of 
    conductance-based networks of spiking neurons with adaptation.
    Neural computation, 31(4):653-690.
"""

from ..base import AbstractDynamics

import jax.numpy as jnp
import jax.scipy as jsp
import jax

from typing import Tuple

from tvboptim.experimental.network_dynamics.core import Bunch

class AdExMF1stOrder(AbstractDynamics):
    """
        Notes
    -----

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('E', 'I', 'W_e', 'W_i', 'noise')`` 
        (exc/inh firing rates [kHz], exc/inh adaptation [pA], noise)
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0., 0., 100., 0., 0.)``
    AUXILIARY_NAMES : tuple of str
        Auxiliary variables: ``(, )``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'delayed': 2}``
    DEFAULT_PARAMS : Bunch
        Standard parameters for AdEx MF exc/inh populations

    References
    ----------
    di Volo et al., 2019. Biologically realistic mean-field models of 
    conductance-based networks of spiking neurons with adaptation.
    Neural computation, 31(4):653-690.
    """

    # Excitation firing rate
    STATE_NAMES = ["E","I","W_e","W_i","noise"]
    INITIAL_STATE = [0.,0.,100.,0.,0.]

    AUXILIARY_NAMES = []

    DEFAULT_PARAMS = Bunch(
        weight_noise = 1e-4, # [kHz] Average of noise input
        tau_OU = 5.0, # [ms] OU process time constant
        
        external_input_ex_to_ex = 0.315*1e-3, # [kHz] exc --> exc constant input
        external_input_in_to_in = 0.000, # [kHz] inh --> exc constant input
        external_input_ex_to_in = 0.315*1e-3, # [kHz] exc --> inh constant input
        external_input_in_to_ex = 0.000, # [kHz] inh --> inh constant input
        
        E_L_e = -63., # [mV] Leak reversal potential for exc pop
        E_L_i = -65., # [mV] Leak reversal potential for inh pop
        g_L_e = 10., # [nS] Leak conductance for exc pop
        g_L_i = 10., # [nS] Leak conductance for inh pop

        C_m = 200.0,  # [pF] Membrane capacitance
        
        b_e = 5.0,  # [pA] Exc adaptation current increment
        a_e = 0.0,  # [nS] Exc adaptation conductance
        b_i = 0.0,  # [pA] Inh adaptation current increment
        a_i = 0.0,  # [nS] Inh adaptation conductance
        
        tau_w_e = 500.0,  # [ms] Adaptation time constant of exc neurons
        tau_w_i = 1.0,  # [ms] Adaptation time constant of inh neurons
        
        E_e = 0.0,  # [mV] Exc reversal potential
        E_i = -80.0,  # [mV] Inh reversal potential
        
        Q_e = 1.5,  # [nS] Exc quantal conductance
        Q_i = 5.0,  # [nS] Inh quantal conductance
        
        tau_e = 5.0,  # [ms] Exc decay
        tau_i = 5.0,  # [ms] Inh decay
        
        N_tot = 10000,  # Total number of neurons
        p_connect_e = 0.05,  # Connectivity probability (exc)
        p_connect_i = 0.05,  # Connectivity probability (inh)
        g = 0.2,  # Fraction of inh neurons
        
        K_ext_e = 400,  # Number of exc connections from external population
        K_ext_i = 0,
        
        T = 20.0,  # [ms] Time scale of network activity
        
        # Fitted polynomial coefficients of the exc transfer function
        P_e = jnp.array([-0.04983106, 0.00506355, -0.02347012, 0.00229515, -0.00041053, 0.00743749, 0.00126506, 0.01054705, -0.04072161, -0.03659253]),
        # Fitted polynomial coefficients of the inh transfer function
        P_i = jnp.array([-0.05149122, 0.00400369, -0.00835201, 0.00024142, -0.00050706, 0.00450271, 0.00284722, 0.00143454, -0.0153578, -0.01468669]),
    )

    COUPLING_INPUTS = {
        "delayed": 2, # Long-range excitation and Feedforward inhibition
    }
        
    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute AdEx mean-field dynamics with two coupling inputs.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[5, n_nodes]`` containing E, I, W_e, W_i, noise
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs with attributes ``.delayed[2, n_nodes]``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[5, n_nodes]``
        """
        E, I, W_e, W_i, noise = state

        # Compute total inputs to each population
        # fx_y: x input to population y
        fe_e, fi_e, fe_i, fi_i = self.inputs_merging(coupling, E, I, noise, params)

        fe_out, (mu_V_e, sig_V_e, tau_V_e) = self.TF_e(fe_e, fi_e, W_e, params)
        fi_out, (mu_V_i, sig_V_i, tau_V_i) = self.TF_i(fe_i, fi_i, W_i, params)

        derivatives = jnp.array([
            (fe_out - E) / params.T,
            (fi_out - I) / params.T,
            -W_e/params.tau_w_e + params.b_e*E + params.a_e*(mu_V_e-params.E_L_e)/params.tau_w_e,
            -W_i/params.tau_w_i + params.b_i*I + params.a_i*(mu_V_i-params.E_L_i)/params.tau_w_i,
            -noise/params.tau_OU])

        return derivatives
    
    def inputs_merging(
        self,
        coupling: Bunch,
        E: jnp.ndarray,
        I: jnp.ndarray,
        noise: jnp.ndarray,
        params: Bunch,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Compute total excitatory and inhibitory inputs.
    
        Parameters
        ----------
        coupling : Bunch
            Coupling inputs with ``coupling.delayed`` of shape
            ``[2, n_nodes]``.
        E : jnp.ndarray
            Excitatory population firing rate with shape ``[n_nodes]``.
        I : jnp.ndarray
            Inhibitory population firing rate with shape ``[n_nodes]``.
        noise : jnp.ndarray
            OU noise with shape ``[n_nodes]``.
        params : Bunch
            Model parameters.
    
        Returns
        -------
        tuple of jnp.ndarray
            ``(fe_e, fi_e, fe_i, fi_i)`` corresponding to:
    
            - excitatory input to the excitatory population,
            - inhibitory input to the excitatory population,
            - excitatory input to the inhibitory population,
            - inhibitory input to the inhibitory population.
        """
        # external exc -> exc input
        fe_ext_e = coupling.delayed[0] + params.weight_noise * noise
        fe_ext_e = (fe_ext_e + params.external_input_ex_to_ex) * params.K_ext_e
        fe_ext_e = jnp.clip(fe_ext_e, 1e-12, jnp.inf)
        # external exc -> inh input
        fe_ext_i = coupling.delayed[1] + params.weight_noise * noise
        fe_ext_i = (fe_ext_i + params.external_input_ex_to_in) * params.K_ext_e
        fe_ext_i = jnp.clip(fe_ext_i, 1e-12, jnp.inf)
        
        # external inh --> exc input (usually 0)
        fi_ext_e = params.external_input_in_to_ex * params.K_ext_i
        # external inh --> inh input (usually 0)
        fi_ext_i = params.external_input_in_to_in * params.K_ext_i

        # local exc input
        fe_local = (E+1.0e-6)*(1.-params.g)*params.p_connect_e*params.N_tot
        # local inh input
        fi_local = (I+1.0e-6)*params.g*params.p_connect_i*params.N_tot
        return fe_local+fe_ext_e, fi_local+fi_ext_e, fe_local+fe_ext_i, fi_local+fi_ext_i
        
    def get_fluct_regime_vars(
        self,
        fe: jnp.ndarray,
        fi: jnp.ndarray,
        W: jnp.ndarray,
        params: Bunch,
        E_L: float,
        g_L: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute membrane-potential statistics in the fluctuation regime.
    
        This implementation follows the mean-field formulation described
        in https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics 
        and computes the mean membrane potential, its standard
        deviation, and the autocorrelation time of the fluctuations.
    
        Parameters
        ----------
        fe : jnp.ndarray
            Excitatory input rate.
        fi : jnp.ndarray
            Inhibitory input rate.
        W : jnp.ndarray
            Adaptation current.
        params : Bunch
            Model parameters.
        E_L : float
            Leak reversal potential.
        delta_mu_V : float
            Shift applied to the mean membrane potential.
        g_L : float
            Leak conductance.
    
        Returns
        -------
        mu_V : jnp.ndarray
            Mean membrane potential.
        sigma_V : jnp.ndarray
            Standard deviation of the membrane-potential fluctuations.
        T_V : jnp.ndarray
            Autocorrelation time constant of the voltage fluctuations.
        """
        # Mean synaptic conductances.
        mu_Ge = params.Q_e * params.tau_e * fe
        mu_Gi = params.Q_i * params.tau_i * fi
    
        # Total membrane conductance and effective membrane time constant.
        mu_G = g_L + mu_Ge + mu_Gi
        mu_G = jnp.maximum(mu_G, 1e-12)
        T_m = params.C_m / mu_G
    
        # Mean membrane potential.
        mu_V = (
            mu_Ge * params.E_e
            + mu_Gi * params.E_i
            + g_L * E_L
            - W
        ) / mu_G
    
        # Amplitude of the postsynaptic membrane-potential response.
        U_e = params.Q_e / mu_G * (params.E_e - mu_V)
        U_i = params.Q_i / mu_G * (params.E_i - mu_V)
    
        # Variance of the membrane-potential fluctuations.
        var_V = (
            fe * (U_e * params.tau_e) ** 2
            / (2.0 * (params.tau_e + T_m))
            + fi * (U_i * params.tau_i) ** 2
            / (2.0 * (params.tau_i + T_m))
        )
        sigma_V = jnp.sqrt(jnp.maximum(var_V, 1e-12))
    
        # Autocorrelation time of the voltage fluctuations.
        T_V_numerator = (
            fe * (U_e * params.tau_e) ** 2
            + fi * (U_i * params.tau_i) ** 2
        )
    
        T_V_denominator = (
            fe * (U_e * params.tau_e) ** 2
            / (params.tau_e + T_m)
            + fi * (U_i * params.tau_i) ** 2
            / (params.tau_i + T_m)
        )
    
        # Protect against division by zero when both input populations
        # contribute negligibly to the fluctuations.
        T_V_denominator = jnp.maximum(T_V_denominator, 1e-12)
        T_V = T_V_numerator / T_V_denominator
        T_V = jnp.maximum(T_V, 1e-12)
    
        return mu_V, sigma_V, T_V
    
    def TF_e(
        self,
        fe: jnp.ndarray,
        fi: jnp.ndarray,
        W: jnp.ndarray,
        params: Bunch,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Compute the excitatory population transfer function."""
        return self.TF(
            fe=fe,
            fi=fi,
            W=W,
            params=params,
            E_L=params.E_L_e,
            P=params.P_e,
            g_L=params.g_L_e,
        )
    
    def TF_i(
        self,
        fe: jnp.ndarray,
        fi: jnp.ndarray,
        W: jnp.ndarray,
        params: Bunch,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Compute the inhibitory population transfer function."""
        return self.TF(
            fe=fe,
            fi=fi,
            W=W,
            params=params,
            E_L=params.E_L_i,
            P=params.P_i,
            g_L=params.g_L_i,
        )
    

    def TF(
        self,
        fe: jnp.ndarray,
        fi: jnp.ndarray,
        W: jnp.ndarray,
        params: Bunch,
        E_L: float,
        P: jnp.ndarray,
        g_L: float,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Compute the neuronal transfer function.
    
        Parameters
        ----------
        fe : jnp.ndarray
            Excitatory input rate.
        fi : jnp.ndarray
            Inhibitory input rate.
        W : jnp.ndarray
            Adaptation current.
        params : Bunch
            Model parameters.
        E_L : float
            Leak reversal potential.
        P : jnp.ndarray
            Polynomial coefficients of the effective threshold approximation.
        g_L : float
            Leak conductance.
    
        Returns
        -------
        f_out : jnp.ndarray
            Estimated firing rate.
        tuple
            Mean membrane potential, voltage standard deviation, and
            voltage autocorrelation time constant.
        """
        mu_V, sigma_V, tau_V = self.get_fluct_regime_vars(
            fe=fe,
            fi=fi,
            W=W,
            params=params,
            E_L=E_L,
            g_L=g_L,
        )
    
        # Effective threshold predicted by the polynomial approximation.
        tau_V_normalized = tau_V * g_L / params.C_m
        V_thre = (
            self.threshold_func(
                muV=mu_V,
                sigmaV=sigma_V,
                TvN=tau_V_normalized,
                P=P,
            )
            * 1e3
        )
    
        # Convert the threshold and voltage fluctuations into a firing rate.
        f_out = self.estimate_firing_rate(
            muV=mu_V,
            sigmaV=sigma_V,
            Tv=tau_V,
            Vthre=V_thre,
        )
    
        return jnp.squeeze(f_out), (mu_V, sigma_V, tau_V)
    
    def threshold_func(
        self,
        muV: jnp.ndarray,
        sigmaV: jnp.ndarray,
        TvN: jnp.ndarray,
        P: jnp.ndarray,
    ) -> jnp.ndarray:
        """Estimate the effective firing threshold from voltage statistics.
    
        Parameters
        ----------
        muV : jnp.ndarray
            Mean membrane potential.
        sigmaV : jnp.ndarray
            Standard deviation of the membrane potential.
        TvN : jnp.ndarray
            Normalized voltage autocorrelation time.
        P : jnp.ndarray
            Polynomial coefficients of the threshold approximation.
    
        Returns
        -------
        jnp.ndarray
            Estimated effective firing threshold.
        """
        # Reference values and scales used to normalize the features.
        muV0, DmuV0 = -60.0, 10.0
        sigmaV0, DsigmaV0 = 4.0, 6.0
        TvN0, DTvN0 = 0.5, 1.0
    
        # Avoid zero-valued bases when differentiating powers with JAX.
        eps = 1e-12
    
        V = (muV - muV0) / DmuV0
        S = (sigmaV - sigmaV0) / DsigmaV0
        T = (TvN - TvN0) / DTvN0
    
        V = jnp.where(jnp.abs(V) < eps, V + eps, V)
        S = jnp.where(jnp.abs(S) < eps, S + eps, S)
        T = jnp.where(jnp.abs(T) < eps, T + eps, T)
    
        # Construct polynomial features.
        V = V.reshape(-1, 1)
        S = S.reshape(-1, 1)
        T = T.reshape(-1, 1)
        
        # Variable exponents for each of the coefficients
        # (order: mu_V, sigma_V, tauN_V)
        exps = jnp.array(
          [[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [2, 0, 0],
           [1, 1, 0],
           [1, 0, 1],
           [0, 2, 0],
           [0, 1, 1],
           [0, 0, 2]])
    
        features = (
            V ** exps[:, 0]
            * S ** exps[:, 1]
            * T ** exps[:, 2]
        )
    
        return features @ P
    

    def estimate_firing_rate(
        self,
        muV: jnp.ndarray,
        sigmaV: jnp.ndarray,
        Tv: jnp.ndarray,
        Vthre: jnp.ndarray,
    ) -> jnp.ndarray:
        """Estimate firing rate from membrane-potential statistics.
    
        Parameters
        ----------
        muV : jnp.ndarray
            Mean membrane potential.
        sigmaV : jnp.ndarray
            Standard deviation of the membrane potential.
        Tv : jnp.ndarray
            Autocorrelation time of the voltage fluctuations.
        Vthre : jnp.ndarray
            Effective firing threshold.
    
        Returns
        -------
        jnp.ndarray
            Estimated firing rate.
        """
        z = (Vthre - muV) / (jnp.sqrt(2.0) * sigmaV)
        return jsp.special.erfc(z) / (2.0 * Tv)

class AdExMF2ndOrder(AdExMF1stOrder):
    """
        Notes
    -----

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('E', 'I', 'C_ee', 'C_ei', 'C_ii', 'W_e', 'W_i', 'noise')`` 
        (exc/inh firing rates [kHz], exc/inh covariances [kHz**2], exc/inh adaptation [pA], noise)
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0., 0., 0., 0., 0., 100., 0., 0.)``
    AUXILIARY_NAMES : tuple of str
        Auxiliary variables: ``(, )``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'delayed': 2}``
    DEFAULT_PARAMS : Bunch
        Standard parameters for AdEx MF exc/inh populations

    References
    ----------
    di Volo et al., 2019. Biologically realistic mean-field models of 
    conductance-based networks of spiking neurons with adaptation.
    Neural computation, 31(4):653-690.
    """
    STATE_NAMES = ["E","I","C_ee","C_ei","C_ii","W_e","W_i","noise"]
    INITIAL_STATE = [0.,0.,0.,0.,0.,100.,0.,0.]

    def __init__(self):
        super().__init__()
        self.dTF_e_dfe = jax.vmap(jax.grad(self.TF_e, argnums=0, has_aux=True), in_axes=(0,0,0,None))
        self.dTF_e_dfi = jax.vmap(jax.grad(self.TF_e, argnums=1, has_aux=True), in_axes=(0,0,0,None))
        self.dTF_i_dfe = jax.vmap(jax.grad(self.TF_i, argnums=0, has_aux=True), in_axes=(0,0,0,None))
        self.dTF_i_dfi = jax.vmap(jax.grad(self.TF_i, argnums=1, has_aux=True), in_axes=(0,0,0,None))
        
        self.d2TF_e = jax.vmap(jax.hessian(self.TF_e, argnums=(0,1), has_aux=True), in_axes=(0,0,0,None))
        self.d2TF_i = jax.vmap(jax.hessian(self.TF_i, argnums=(0,1), has_aux=True), in_axes=(0,0,0,None))

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute second-order AdEx mean-field dynamics with two coupling inputs.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[8, n_nodes]`` containing E, I, C_ee, C_ei, C_ii, W_e, W_i, noise
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs with attributes ``.delayed[2, n_nodes]``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[8, n_nodes]``
        """        

        N_e = params.N_tot * (1 - params.g)
        N_i = params.N_tot * params.g
        
        E, I, C_ee, C_ei, C_ii, W_e, W_i, noise = state

        # fx_y: x input to population y
        fe_e, fi_e, fe_i, fi_i = self.inputs_merging(coupling, E, I, noise, params)     

        fe_out, (mu_V_e, sig_V_e, tau_V_e) = self.TF_e(fe_e, fi_e, W_e, params)
        fi_out, (mu_V_i, sig_V_i, tau_V_i) = self.TF_i(fe_i, fi_i, W_i, params)

        d2TF_e_values = self.d2TF_e(fe_e, fi_e, W_e, params)[0]
        d2TF_i_values = self.d2TF_i(fe_i, fi_i, W_i, params)[0]
        dTF_e_dfe_value = self.dTF_e_dfe(fe_e, fi_e, W_e, params)[0]
        dTF_e_dfi_value = self.dTF_e_dfi(fe_e, fi_e, W_e, params)[0]
        dTF_i_dfe_value = self.dTF_i_dfe(fe_i, fi_i, W_i, params)[0]
        dTF_i_dfi_value = self.dTF_i_dfi(fe_i, fi_i, W_i, params)[0]
        
        diff_e = fe_out - E
        diff_i = fi_out - I
        
        derivatives = jnp.array([
            (diff_e + .5*(C_ee*d2TF_e_values[0][0] + C_ei*(d2TF_e_values[0][1] + d2TF_e_values[1][0]) + C_ii*d2TF_e_values[1][1]))/params.T, # E
            (diff_i + .5*(C_ee*d2TF_i_values[0][0] + C_ei*(d2TF_i_values[0][1] + d2TF_i_values[1][0]) + C_ii*d2TF_i_values[1][1]))/params.T, # I
            (fe_out*(params.T**-1-fe_out)/N_e + diff_e**2 + 2.*(C_ee*dTF_e_dfe_value + C_ei*dTF_e_dfi_value - C_ee))/params.T, # C_ee
            (diff_e*diff_i + C_ee*dTF_e_dfe_value + C_ei*(dTF_i_dfe_value + dTF_e_dfi_value) + C_ii*dTF_i_dfi_value - 2.*C_ei)/params.T, # C_ie
            (fi_out*(params.T**-1-fi_out)/N_i + diff_i**2 + 2.*(C_ii*dTF_i_dfi_value + C_ei*dTF_i_dfe_value - C_ii))/params.T, # C_ii
            -W_e/params.tau_w_e + params.b_e*E + params.a_e*(mu_V_e-params.E_L_e)/params.tau_w_e, # W_e
            -W_i/params.tau_w_i + params.b_i*I + params.a_i*(mu_V_i-params.E_L_i)/params.tau_w_i, # w_i
            -noise/params.tau_OU 
            ])
        return derivatives