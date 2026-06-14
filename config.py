"""
Configuration parameters for the Digital Twin Stochastic MPC framework
"""

import numpy as np

# Simulation parameters
SAMPLING_TIME = 5  # minutes
SIMULATION_DAYS = 1  # days per scenario
WARMUP_HOURS = 4  # hours for initialization

# Glucose targets and bounds
TARGET_GLUCOSE = 115  # mg/dL
G_MIN = 70  # mg/dL - hypoglycemia threshold
G_MAX = 250  # mg/dL - severe hyperglycemia threshold
G_SEVERE_HYPO = 54  # mg/dL - severe hypoglycemia
G_GUARD = 90  # mg/dL - soft guardrail to discourage low-glucose drift
TIR_LOWER = 70  # mg/dL
TIR_UPPER = 180  # mg/dL

# MPC parameters
MPC_HORIZON = 12  # steps (60 minutes with 5-min sampling)
MPC_CONTROL_INTERVAL = SAMPLING_TIME  # minutes
U_MAX = 0.1  # U/min - maximum insulin infusion rate
U_MIN = 0.0  # U/min - minimum insulin infusion rate
DELTA_U_MAX = 0.02  # U/min - maximum rate of change

# Stochastic MPC parameters
EPSILON_HYPO = 0.01  # 1% risk tolerance for hypoglycemia
EPSILON_HYPER = 0.10  # 10% risk tolerance for hyperglycemia
KAPPA_HYPO = 2.326  # z-score for 1% one-sided risk (Φ^{-1}(0.99))
KAPPA_HYPER = 1.282  # z-score for 10% one-sided risk (Φ^{-1}(0.90))

# Cost function weights
W_GLUCOSE = 1.0  # weight for glucose tracking error
W_VARIANCE = 0.5  # weight for predictive variance
W_INSULIN = 0.01  # weight for insulin usage
W_DELTA_U = 0.1  # weight for insulin rate changes
W_RISK_SLACK = 200.0  # penalty on chance-constraint violations
W_LOW_GLUCOSE = 5.0  # penalty weight for predicted low-glucose excursions

# SDE model parameters
THETA_SENSITIVITY = 0.08  # mean reversion rate for insulin sensitivity
SIGMA_SENSITIVITY = 0.08  # volatility of insulin sensitivity
SIGMA_MEAL = 0.15  # meal absorption noise
SIGMA_BASAL = 0.05  # basal glucose dynamics noise
# Insulin compartment noise: after the basal-calibration fix raised X_b from 10 to
# ~30-40 mU/L, the former PROCESS_NOISE_SCALE=1.0 blanket value caused insulin-action
# (X) to random-walk into regimes where EGP = EGP_b*exp(-0.05*(X-X_b)) could reach
# 7× baseline (X→0).  These targeted values keep per-step σ < 1% of steady-state,
# preventing catastrophic EGP amplification while preserving meaningful SMPC variance.
SIGMA_INSULIN_PLASMA = 0.10   # I_p process noise (mU/L/√min)
SIGMA_INSULIN_SC     = 1.0    # I_sc1, I_sc2 process noise (mU/√min) — large compartments
SIGMA_INSULIN_ACTION = 0.10   # X  process noise (mU/L/√min)

# Kalman filter parameters
PROCESS_NOISE_SCALE = 1.0  # scaling factor for process noise covariance
MEASUREMENT_NOISE_STD = 5.0  # mg/dL - CGM sensor noise
CGM_DELAY_STEPS = 1  # one 5-minute sample of delay in the control loop

# PID controller parameters
PID_GAINS = {
    'adult': {'Kp': 0.00035, 'Ki': 0.00003, 'Kd': 0.00020},
    'adolescent': {'Kp': 0.00040, 'Ki': 0.00003, 'Kd': 0.00025},
    'child': {'Kp': 0.00045, 'Ki': 0.00004, 'Kd': 0.00030},
}

# Meal scenarios — body-weight normalised (g/kg).
# Absolute grams are computed at simulation time by multiplying by patient BW.
# Reference: pediatric T1D meal challenge norms (~0.67–0.93 g/kg for main meals).
MEAL_SCENARIOS_GPKG = {
    'standard': {
        'breakfast':  {'time': 7*60,  'cho_gpkg': 0.67, 'announce': True},   # ~50 g for 75 kg adult
        'lunch':      {'time': 12*60, 'cho_gpkg': 0.93, 'announce': True},   # ~70 g for 75 kg adult
        'dinner':     {'time': 18*60, 'cho_gpkg': 0.80, 'announce': True},   # ~60 g for 75 kg adult
    },
    'variable': {
        'breakfast': {'time_range': (6*60, 8*60),   'cho_gpkg_range': (0.40, 1.20)},
        'lunch':     {'time_range': (11*60, 13*60), 'cho_gpkg_range': (0.40, 1.20)},
        'dinner':    {'time_range': (17*60, 19*60), 'cho_gpkg_range': (0.40, 1.20)},
        'snack':     {'probability': 0.3, 'time_range': (14*60, 16*60),
                      'cho_gpkg_range': (0.13, 0.27), 'announce': False},
    },
    'nocturnal': {
        'breakfast':  {'time': 7*60,  'cho_gpkg': 0.67, 'announce': True},
        'lunch':      {'time': 12*60, 'cho_gpkg': 0.93, 'announce': True},
        'dinner':     {'time': 18*60, 'cho_gpkg': 0.80, 'announce': False},
        'late_snack': {'time': 22*60, 'cho_gpkg': 0.24, 'announce': False},  # ~18 g for 75 kg adult
    },
}

# Legacy absolute-gram scenario dict — retained so that build_manuscript_assets.py
# iteration over config.MEAL_SCENARIOS continues to work unchanged.
MEAL_SCENARIOS = {
    'standard': {
        'breakfast': {'time': 7*60, 'cho': 50},
        'lunch':     {'time': 12*60, 'cho': 70},
        'dinner':    {'time': 18*60, 'cho': 60}
    },
    'variable': {
        'breakfast': {'time_range': (6*60, 8*60),   'cho_range': (30, 90)},
        'lunch':     {'time_range': (11*60, 13*60), 'cho_range': (30, 90)},
        'dinner':    {'time_range': (17*60, 19*60), 'cho_range': (30, 90)},
        'snack': {'probability': 0.3, 'time_range': (14*60, 16*60),
                  'cho_range': (10, 20), 'announce': False}
    },
    'nocturnal': {
        'breakfast':  {'time': 7*60,  'cho': 50},
        'lunch':      {'time': 12*60, 'cho': 70},
        'dinner':     {'time': 18*60, 'cho': 60,  'announce': False},
        'late_snack': {'time': 22*60, 'cho': 18,  'announce': False}
    }
}

# Virtual patient selection (synthetic S2017)
SUBJECTS_PER_COHORT = 100  # matches UVA/Padova S2017 public specs
COHORTS = ['adult', 'adolescent', 'child']
NUM_PATIENTS = SUBJECTS_PER_COHORT  # per-cohort loop length
PADOVA_SEED = 42  # reproducible synthetic cohort

# Hyperparameter tuning train/test partition
# First 30 adult subjects (indices 0-29, seed 42) were used exclusively for
# grid-search tuning of SMPC, DMPC, and PID weights.  The remaining 70 adult
# subjects and all 200 non-adult subjects form the held-out test set.
# See code/train_test_split.py and code/tune_hyperparameters.py.
TUNING_COHORT   = 'adult'
TUNING_SCENARIO = 'standard'
TRAIN_FRACTION  = 0.30

# Output directories
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'
DATA_DIR = 'results/data'
