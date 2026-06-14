"""
Digital Twin model based on stochastic differential equations
Extends the deterministic UVA/Padova model with uncertainty
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Optional
import config


class StochasticDigitalTwin:
    """
    Digital twin with SDE representation of glucose-insulin dynamics
    """
    
    def __init__(self, patient_params: Dict, dt: float = config.SAMPLING_TIME):
        """
        Initialize digital twin for a specific patient
        
        Args:
            patient_params: Dictionary of patient-specific parameters
            dt: Time step in minutes
        """
        self.patient_params = patient_params
        self.dt = dt  # minutes
        self.dt_hours = dt / 60.0  # hours
        
        # State dimension (simplified model)
        # States: [G_p, G_i, I_p, I_sc1, I_sc2, X, q_sto1, q_sto2, q_gut]
        # G_p: plasma glucose (mg/dL)
        # G_i: interstitial glucose (mg/dL)
        # I_p: plasma insulin (mU/L)
        # I_sc1: subcutaneous insulin compartment 1 (mU)
        # I_sc2: subcutaneous insulin compartment 2 (mU)
        # X: insulin action (mU/L) - remote compartment
        # q_sto1: stomach solid (mg)
        # q_sto2: stomach liquid (mg)
        # q_gut: intestine (mg)
        self.n_states = 9
        
        # Initialize state
        self.state = np.zeros(self.n_states)
        self.state[0] = 120.0  # plasma glucose (mg/dL)
        self.state[1] = 120.0  # interstitial glucose (mg/dL)
        self.state[2] = 10.0   # plasma insulin (mU/L)
        self.state[5] = 10.0   # X (mU/L) - steady state
        
        # Stochastic components
        self.insulin_sensitivity_multiplier = 1.0
        
        # Process noise covariance
        self.Q = self._build_process_noise_covariance()
        
        # Measurement noise covariance
        self.R = config.MEASUREMENT_NOISE_STD**2
        
    def _build_process_noise_covariance(self) -> np.ndarray:
        """Build process noise covariance matrix"""
        Q = np.eye(self.n_states) * config.PROCESS_NOISE_SCALE
        Q[0, 0] = config.SIGMA_BASAL**2           # G_p  plasma glucose
        Q[1, 1] = config.SIGMA_BASAL**2           # G_i  interstitial glucose
        Q[2, 2] = config.SIGMA_INSULIN_PLASMA**2  # I_p  plasma insulin
        Q[3, 3] = config.SIGMA_INSULIN_SC**2      # I_sc1 subcutaneous compartment 1
        Q[4, 4] = config.SIGMA_INSULIN_SC**2      # I_sc2 subcutaneous compartment 2
        Q[5, 5] = config.SIGMA_INSULIN_ACTION**2  # X    insulin action (remote)
        Q[6, 6] = config.SIGMA_MEAL**2            # q_sto1 meal absorption
        Q[7, 7] = config.SIGMA_MEAL**2            # q_sto2
        Q[8, 8] = config.SIGMA_MEAL**2            # q_gut
        return Q
    
    def dynamics(self, state: np.ndarray, u: float, meal_cho: float = 0.0) -> np.ndarray:
        """
        Deterministic part of state dynamics (continuous time)
        
        Args:
            state: Current state vector
            u: Insulin infusion rate (U/min)
            meal_cho: Carbohydrate intake (g)
        
        Returns:
            Time derivative of state
        """
        G_p, G_i, I_p, I_sc1, I_sc2, X, q_sto1, q_sto2, q_gut = state
        
        # Patient parameters
        BW = self.patient_params.get('BW', 75.0)  # body weight (kg)
        V_G = 1.88  # glucose distribution volume (dL/kg)
        V_I = 0.05  # insulin distribution volume (L/kg)
        
        # Constants
        k_12 = 0.066  # transfer rate (1/min)
        k_21 = 0.066
        k_i = 0.0079  # insulin clearance
        k_e = 0.138   # elimination (corrected from 0.0071)
        k_a1 = 0.006  # insulin absorption
        k_a2 = 0.06
        k_a = 0.006   # insulin action
        k_b = 0.006
        
        SI = self.patient_params.get('SI', 0.0001) * self.insulin_sensitivity_multiplier

        # Patient-specific basal steady state — compute from actual basal_rate so that
        # EGP calibration balances insulin action at the true pharmacokinetic equilibrium.
        # (Hardcoding X_b=10 caused glucose drift because real steady-state X is 3-4x higher.)
        G_b = 120.0
        basal_U_hr = self.patient_params.get('basal_rate', 1.0)
        u_mU_b = basal_U_hr / 60.0 * 1000.0  # mU/min at basal infusion
        X_b = u_mU_b / ((k_i + k_e) * V_I * BW)  # SS X = SS I_p since k_a = k_b
        
        # Fluxes
        U_ii_flux = 1.0 * BW # mg/min (CNS utilization)
        U_ii = U_ii_flux / (V_G * BW) # mg/dL/min
        
        # Insulin Dependent Utilization
        # U_id = SI * X * G_p (mg/dL/min)
        U_id = SI * X * G_p
        
        # Endogenous Glucose Production
        # Calibrate EGP_b to balance utilization at steady state
        U_id_b = SI * X_b * G_b
        EGP_b = U_ii + U_id_b # mg/dL/min
        
        # EGP suppression model — relative formulation so the suppression ratio
        # is invariant to patient-specific X_b.  Using exp(-0.5*(X/X_b - 1))
        # reproduces the old exp(-0.05*(X-10)) suppression ratios at relative X
        # values (0x, 1x, 2x of X_b) regardless of how large X_b is.  This
        # prevents the 7× EGP spike that occurred when X_b was corrected from
        # 10 to ~39 mU/L and X occasionally drained to 0 with u=0.
        X_b_safe = max(X_b, 1e-6)
        EGP = EGP_b * np.exp(-0.5 * (X / X_b_safe - 1.0))
        EGP = max(0.0, EGP)
        
        # Meal Appearance
        # q_gut is in mg
        k_abs = 0.012
        f_abs = 0.9
        Ra_flux = f_abs * k_abs * q_gut # mg/min
        Ra = Ra_flux / (V_G * BW) # mg/dL/min
        
        # Glucose Dynamics
        dG_p = EGP + Ra - U_ii - U_id - k_12 * G_p + k_21 * G_i
        dG_i = k_12 * G_p - k_21 * G_i
        
        # Insulin Dynamics
        # u is U/min -> * 1e6 mU/min? No, 1 U = 1000 mU.
        u_mU = u * 1000.0
        
        dI_sc1 = u_mU - k_a1 * I_sc1
        dI_sc2 = k_a1 * I_sc1 - k_a2 * I_sc2
        
        # Plasma insulin appearance from SC
        Ra_I = k_a2 * I_sc2 # mU/min
        Ra_I_conc = Ra_I / (V_I * BW) # mU/L/min
        
        # Plasma insulin clearance
        # dI_p = -k_cl * I_p + Ra_I_conc
        k_cl = k_i + k_e
        dI_p = -k_cl * I_p + Ra_I_conc
        
        # Insulin Action
        dX = -k_b * X + k_a * I_p
        
        # Stomach Dynamics
        # Meal input is handled in step(), here we just do emptying
        k_max = 0.0558
        k_min = 0.008
        D = 1000.0
        
        q_sto_total = q_sto1 + q_sto2
        kempt = k_min + (k_max - k_min) / 2 * (1 - np.tanh(2 * (q_sto_total - D) / D))
            
        dq_sto1 = -kempt * q_sto1
        dq_sto2 = kempt * q_sto1 - k_abs * q_sto2
        dq_gut = k_abs * q_sto2 - k_abs * q_gut
        
        return np.array([dG_p, dG_i, dI_p, dI_sc1, dI_sc2, dX, dq_sto1, dq_sto2, dq_gut])
    
    def step(self, u: float, meal_cho: float = 0.0, add_noise: bool = True) -> Tuple[np.ndarray, float]:
        """
        Advance state by one time step using Euler-Maruyama method
        
        Args:
            u: Insulin infusion rate (U/min)
            meal_cho: Carbohydrate intake (g)
            add_noise: Whether to add stochastic noise
        
        Returns:
            New state and glucose measurement
        """
        # Reduced-order surrogate shortcut: split CHO across solid and liquid stomach states.
        # Full Padova-style meal handling would load the gastric solid compartment first.
        if meal_cho > 0:
            self.state[6] += meal_cho * 1000 * 0.5  # 50% to first compartment
            self.state[7] += meal_cho * 1000 * 0.5  # 50% to second compartment
        
        # Update insulin sensitivity (Ornstein-Uhlenbeck process)
        if add_noise:
            dW_sensitivity = np.random.randn() * np.sqrt(self.dt)
            self.insulin_sensitivity_multiplier += config.THETA_SENSITIVITY * (1.0 - self.insulin_sensitivity_multiplier) * self.dt
            self.insulin_sensitivity_multiplier += config.SIGMA_SENSITIVITY * dW_sensitivity
            self.insulin_sensitivity_multiplier = max(0.5, min(1.5, self.insulin_sensitivity_multiplier))
        
        # Deterministic dynamics
        f = self.dynamics(self.state, u, 0.0)  # meal already added
        
        # Euler-Maruyama step
        self.state += f * self.dt
        
        # Add process noise
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n_states), self.Q * self.dt)
            self.state += noise
        
        # Ensure non-negative states
        self.state = np.maximum(self.state, 0.0)
        
        # Measurement (interstitial glucose with noise)
        y = self.state[1]  # Interstitial glucose
        if add_noise:
            y += np.random.randn() * config.MEASUREMENT_NOISE_STD
        
        return self.state.copy(), y
    
    def predict_trajectory(self, u_sequence: np.ndarray, initial_state: np.ndarray, 
                          initial_cov: np.ndarray, meal_schedule: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance of glucose trajectory over horizon
        
        Args:
            u_sequence: Control sequence [N]
            initial_state: Initial state mean [n_states]
            initial_cov: Initial state covariance [n_states x n_states]
            meal_schedule: Dictionary mapping time steps to meal CHO amounts
        
        Returns:
            glucose_mean: Predicted glucose mean [N]
            glucose_var: Predicted glucose variance [N]
        """
        N = len(u_sequence)
        glucose_mean = np.zeros(N)
        glucose_var = np.zeros(N)
        
        state_mean = initial_state.copy()
        state_cov = initial_cov.copy()
        
        for k in range(N):
            # Meal at this step
            meal_cho = 0.0
            if meal_schedule and k in meal_schedule:
                meal_cho = meal_schedule[k]
            
            # Linearize dynamics around current mean
            A, B = self._linearize_dynamics(state_mean, u_sequence[k])
            
            # Predict state mean
            f = self.dynamics(state_mean, u_sequence[k], meal_cho)
            state_mean = state_mean + f * self.dt
            if meal_cho > 0:
                state_mean[6] += meal_cho * 1000 * 0.5
                state_mean[7] += meal_cho * 1000 * 0.5
            
            # Predict state covariance (EKF propagation)
            state_cov = A @ state_cov @ A.T + self.Q * self.dt
            
            # Extract glucose mean and variance
            glucose_mean[k] = state_mean[1]  # Interstitial glucose
            glucose_var[k] = state_cov[1, 1] + self.R  # Include measurement noise
        
        return glucose_mean, glucose_var
    
    def _linearize_dynamics(self, state: np.ndarray, u: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around current state using finite differences
        
        Returns:
            A: State Jacobian [n_states x n_states]
            B: Control Jacobian [n_states x 1]
        """
        epsilon = 1e-6
        
        # Compute A matrix
        A = np.eye(self.n_states)
        f0 = self.dynamics(state, u, 0.0)
        for i in range(self.n_states):
            state_pert = state.copy()
            state_pert[i] += epsilon
            f_pert = self.dynamics(state_pert, u, 0.0)
            A[:, i] = A[:, i] + (f_pert - f0) / epsilon * self.dt
        
        # Compute B matrix
        f_u0 = self.dynamics(state, u, 0.0)
        f_u1 = self.dynamics(state, u + epsilon, 0.0)
        B = ((f_u1 - f_u0) / epsilon * self.dt).reshape(-1, 1)
        
        return A, B
    
    def get_measurement(self) -> float:
        """Get current glucose measurement"""
        return self.state[1]  # Interstitial glucose
    
    def reset(self, initial_glucose: float = 120.0):
        """Reset digital twin to initial state with variability"""
        self.state = np.zeros(self.n_states)
        # Add some variability to initial glucose
        self.state[0] = initial_glucose + np.random.randn() * 10
        self.state[1] = initial_glucose + np.random.randn() * 10
        self.state[0] = np.clip(self.state[0], 80, 180)
        self.state[1] = np.clip(self.state[1], 80, 180)
        
        # Initialize insulin compartments to patient-specific basal steady state
        _k_a1, _k_a2 = 0.006, 0.06
        _k_cl = 0.0079 + 0.138
        _V_I = 0.05
        _BW = self.patient_params.get('BW', 75.0)
        _u_mU_b = self.patient_params.get('basal_rate', 1.0) / 60.0 * 1000.0
        _I_p_ss = _u_mU_b / (_k_cl * _V_I * _BW)
        self.state[2] = _I_p_ss * (1.0 + np.random.randn() * 0.05)   # I_p ±5%
        self.state[3] = _u_mU_b / _k_a1                               # I_sc1
        self.state[4] = _u_mU_b / _k_a2                               # I_sc2
        self.state[5] = _I_p_ss * (1.0 + np.random.randn() * 0.05)   # X ≈ I_p at SS
        
        self.insulin_sensitivity_multiplier = 1.0 + np.random.randn() * 0.1
