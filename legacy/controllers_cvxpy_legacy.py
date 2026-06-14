"""
Stochastic Model Predictive Control with chance constraints
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional, Dict
from digital_twin import StochasticDigitalTwin
import config


class StochasticMPC:
    """
    Chance-constrained stochastic MPC for glucose control
    """
    
    def __init__(self, digital_twin: StochasticDigitalTwin):
        """
        Initialize stochastic MPC controller
        
        Args:
            digital_twin: Digital twin model for prediction
        """
        self.dt_model = digital_twin
        self.N = config.MPC_HORIZON
        self.dt = config.SAMPLING_TIME
        
        # Cost function weights
        self.w_glucose = config.W_GLUCOSE
        self.w_variance = config.W_VARIANCE
        self.w_insulin = config.W_INSULIN
        self.w_delta_u = config.W_DELTA_U
        
        # Glucose targets and bounds
        self.G_target = config.TARGET_GLUCOSE
        self.G_min = config.G_MIN
        self.G_max = config.G_MAX
        
        # Chance constraint parameters
        self.kappa_hypo = config.KAPPA_HYPO
        self.kappa_hyper = config.KAPPA_HYPER
        
        # Control bounds
        self.u_min = config.U_MIN
        self.u_max = config.U_MAX
        self.delta_u_max = config.DELTA_U_MAX
        
        # Previous control
        self.u_prev = 0.0
        
    def solve(self, state_mean: np.ndarray, state_cov: np.ndarray, 
              meal_schedule: Optional[Dict] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve stochastic MPC optimization problem (simplified version)
        
        Args:
            state_mean: Current state estimate (mean)
            state_cov: Current state covariance
            meal_schedule: Dictionary mapping time steps to meal CHO
        
        Returns:
            optimal_control: First control action
            u_sequence: Full optimal control sequence
            glucose_mean: Predicted glucose mean trajectory
            glucose_std: Predicted glucose standard deviation trajectory
        """
        # Current glucose
        current_glucose = state_mean[1]  # interstitial glucose
        glucose_std = np.sqrt(state_cov[1, 1])
        
        # Simple control law based on glucose and uncertainty
        # If glucose is high and uncertainty is low, increase insulin
        # If glucose is low or uncertainty is high, reduce insulin
        
        error = current_glucose - self.G_target
        
        # Base control proportional to error
        u_base = 0.001 * error
        
        # Adjust for hypoglycemia risk (conservative)
        lower_bound = current_glucose - self.kappa_hypo * glucose_std
        if lower_bound < self.G_min:
            # High hypo risk - reduce insulin
            u_base *= 0.5
        
        # Apply bounds
        u_optimal = np.clip(u_base, self.u_min, self.u_max)
        
        # Rate limit
        delta_u = u_optimal - self.u_prev
        if abs(delta_u) > self.delta_u_max:
            u_optimal = self.u_prev + np.sign(delta_u) * self.delta_u_max
        
        self.u_prev = u_optimal
        
        # Create sequence (simplified - constant control)
        u_sequence = np.ones(self.N) * u_optimal
        
        # Predict trajectory
        glucose_mean, glucose_var = self.dt_model.predict_trajectory(
            u_sequence, state_mean, state_cov, meal_schedule
        )
        glucose_std = np.sqrt(glucose_var)
        
        return u_optimal, u_sequence, glucose_mean, glucose_std
    
    def reset(self):
        """Reset controller state"""
        self.u_prev = 0.0


class DeterministicMPC:
    """
    Deterministic MPC baseline (ignores uncertainty)
    """
    
    def __init__(self, digital_twin: StochasticDigitalTwin):
        """
        Initialize deterministic MPC
        
        Args:
            digital_twin: Digital twin for predictions
        """
        self.dt_model = digital_twin
        self.N = config.MPC_HORIZON
        self.w_glucose = config.W_GLUCOSE
        self.w_insulin = config.W_INSULIN
        self.w_delta_u = config.W_DELTA_U
        self.G_target = config.TARGET_GLUCOSE
        self.G_min = config.G_MIN
        self.G_max = config.G_MAX
        self.u_min = config.U_MIN
        self.u_max = config.U_MAX
        self.delta_u_max = config.DELTA_U_MAX
        self.u_prev = 0.0
    
    def solve(self, state_mean: np.ndarray, meal_schedule: Optional[Dict] = None) -> float:
        """
        Solve deterministic MPC (simplified version with direct control law)
        
        Args:
            state_mean: Current state estimate
            meal_schedule: Announced meals
        
        Returns:
            optimal_control: Control action
        """
        # Current glucose
        current_glucose = state_mean[1]  # interstitial glucose
        
        # Simple proportional control with tighter constraints than PID
        error = current_glucose - self.G_target
        
        u_optimal = 0.0015 * error  # Slightly more aggressive than stochastic MPC
        
        # Hard glucose bounds (be conservative)
        if current_glucose < self.G_min + 10:
            u_optimal = 0.0  # Stop insulin if near hypo
        elif current_glucose > self.G_max - 20:
            u_optimal = max(u_optimal, 0.05)  # Ensure sufficient insulin
        
        # Apply bounds
        u_optimal = np.clip(u_optimal, self.u_min, self.u_max)
        
        # Rate limit
        delta_u = u_optimal - self.u_prev
        if abs(delta_u) > self.delta_u_max:
            u_optimal = self.u_prev + np.sign(delta_u) * self.delta_u_max
        
        self.u_prev = u_optimal
        return u_optimal
    
    def reset(self):
        """Reset controller"""
        self.u_prev = 0.0


class PIDController:
    """
    PID controller baseline
    """
    
    def __init__(self):
        """Initialize PID controller"""
        self.Kp = config.KP
        self.Ki = config.KI
        self.Kd = config.KD
        self.G_target = config.TARGET_GLUCOSE
        self.u_min = config.U_MIN
        self.u_max = config.U_MAX
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = config.SAMPLING_TIME
        
    def solve(self, glucose: float) -> float:
        """
        Compute PID control action
        
        Args:
            glucose: Current glucose measurement
        
        Returns:
            control: Insulin infusion rate
        """
        error = glucose - self.G_target
        
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Saturation
        u = np.clip(u, self.u_min, self.u_max)
        
        # Anti-windup
        if u >= self.u_max or u <= self.u_min:
            self.integral -= error * self.dt
        
        self.prev_error = error
        
        return u
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
