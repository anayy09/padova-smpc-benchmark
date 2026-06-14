"""
Extended Kalman Filter for state estimation
"""

import numpy as np
from typing import Tuple
from digital_twin import StochasticDigitalTwin
import config


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for glucose-insulin state estimation
    """
    
    def __init__(self, digital_twin: StochasticDigitalTwin):
        """
        Initialize EKF
        
        Args:
            digital_twin: Digital twin model for dynamics
        """
        self.dt_model = digital_twin
        self.n_states = digital_twin.n_states
        
        # Initialize from the digital twin's current state so the EKF starts
        # consistent with the patient-specific basal steady state (I_p, I_sc, X).
        # Initializing at zero caused catastrophic insulin over-delivery after the
        # basal calibration fix raised I_p from ~10 to ~30-40 mU/L.
        self.state_mean = digital_twin.state.copy()

        self.state_cov = np.eye(self.n_states) * 100.0  # Initial uncertainty
        
        # Process and measurement noise
        self.Q = digital_twin.Q
        self.R = digital_twin.R
        
    def predict(self, u: float, meal_cho: float = 0.0):
        """
        Prediction step
        
        Args:
            u: Control input (insulin infusion)
            meal_cho: Announced meal CHO
        """
        # Linearize around current estimate
        A, B = self.dt_model._linearize_dynamics(self.state_mean, u)
        
        # Predict state mean
        f = self.dt_model.dynamics(self.state_mean, u, meal_cho)
        self.state_mean = self.state_mean + f * self.dt_model.dt
        
        if meal_cho > 0:
            self.state_mean[6] += meal_cho * 1000 * 0.5
            self.state_mean[7] += meal_cho * 1000 * 0.5
        
        self.state_mean = np.maximum(self.state_mean, 0.0)
        
        # Predict state covariance
        self.state_cov = A @ self.state_cov @ A.T + self.Q * self.dt_model.dt
        
    def update(self, measurement: float):
        """
        Update step with glucose measurement
        
        Args:
            measurement: CGM glucose measurement (mg/dL)
        """
        # Measurement model: y = G_i (interstitial glucose)
        H = np.zeros((1, self.n_states))
        H[0, 1] = 1.0  # Observe interstitial glucose
        
        # Innovation
        y_pred = H @ self.state_mean
        innovation = measurement - y_pred[0]
        
        # Innovation covariance
        S = H @ self.state_cov @ H.T + self.R
        
        # Kalman gain
        K = self.state_cov @ H.T / S[0, 0]
        
        # Update state estimate
        # Ensure K is flattened to avoid broadcasting issues (n,) + (n,1) -> (n,n)
        self.state_mean = self.state_mean + K.flatten() * innovation
        self.state_mean = np.maximum(self.state_mean, 0.0)
        
        # Update covariance
        I_KH = np.eye(self.n_states) - np.outer(K, H)
        self.state_cov = I_KH @ self.state_cov @ I_KH.T + np.outer(K, K) * self.R
        
        # Ensure symmetry and positive definiteness
        self.state_cov = (self.state_cov + self.state_cov.T) / 2
        
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate
        
        Returns:
            state_mean: State mean vector
            state_cov: State covariance matrix
        """
        return self.state_mean.copy(), self.state_cov.copy()
    
    def reset(self, initial_glucose: float = 120.0):
        """Reset filter — sync from the digital twin's current state."""
        self.state_mean = self.dt_model.state.copy()
        self.state_mean[0] = initial_glucose
        self.state_mean[1] = initial_glucose
        self.state_cov = np.eye(self.n_states) * 100.0
