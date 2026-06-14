"""
Simulation scenarios and patient cohort management
"""

import numpy as np
from typing import Dict, List, Tuple
import config
from padova_synthetic import generate_population


PADOVA_POPULATION = generate_population(seed=config.PADOVA_SEED)


def generate_patient_parameters(patient_id: int, cohort: str = 'adult') -> Dict:
    """
    Generate patient-specific parameters
    
    Args:
        patient_id: Patient identifier
        cohort: 'adult', 'adolescent', or 'child'
    
    Returns:
        Dictionary of patient parameters
    """
    subjects = PADOVA_POPULATION.get(cohort)
    if subjects is None:
        raise ValueError(f"Unknown cohort '{cohort}'. Expected one of {list(PADOVA_POPULATION.keys())}.")

    subject = subjects[patient_id % len(subjects)].copy()
    subject['patient_id'] = patient_id
    subject.setdefault('basal_rate', 1.0)
    subject.setdefault('CR', 12.0)
    subject.setdefault('ISF', 50.0)

    return subject


def generate_meal_scenario(
    scenario_type: str, day: int = 0, patient_bw: float = 75.0
) -> Tuple[Dict[int, float], Dict[int, bool]]:
    """
    Generate meal schedule for simulation using body-weight-normalised CHO (g/kg).

    Args:
        scenario_type: 'standard', 'variable', or 'nocturnal'
        day: Day number for randomization seed
        patient_bw: Patient body weight in kg (scales g/kg meal sizes to grams)

    Returns:
        Tuple of (meal_schedule, meal_announcements)
        meal_schedule: Dict mapping time step to CHO amount (grams)
        meal_announcements: Dict mapping time step to announcement flag
    """
    np.random.seed(day)
    meal_schedule: Dict[int, float] = {}
    meal_announcements: Dict[int, bool] = {}
    scenarios = config.MEAL_SCENARIOS_GPKG[scenario_type]

    if scenario_type == 'standard':
        for meal_info in scenarios.values():
            time_step = meal_info['time'] // config.SAMPLING_TIME
            meal_schedule[time_step] = meal_info['cho_gpkg'] * patient_bw
            meal_announcements[time_step] = meal_info.get('announce', True)

    elif scenario_type == 'variable':
        for meal_name in ['breakfast', 'lunch', 'dinner']:
            meal_info = scenarios[meal_name]
            time_min, time_max = meal_info['time_range']
            gpkg_min, gpkg_max = meal_info['cho_gpkg_range']
            time_step = np.random.randint(time_min, time_max) // config.SAMPLING_TIME
            meal_schedule[time_step] = np.random.uniform(gpkg_min, gpkg_max) * patient_bw
            meal_announcements[time_step] = True
        # Occasional unannounced snack
        snack = scenarios['snack']
        if np.random.rand() < snack['probability']:
            snack_time = np.random.randint(*snack['time_range'])
            snack_cho = np.random.uniform(*snack['cho_gpkg_range']) * patient_bw
            time_step = snack_time // config.SAMPLING_TIME
            meal_schedule[time_step] = meal_schedule.get(time_step, 0.0) + snack_cho
            meal_announcements[time_step] = snack.get('announce', False)

    elif scenario_type == 'nocturnal':
        for meal_info in scenarios.values():
            time_step = meal_info['time'] // config.SAMPLING_TIME
            meal_schedule[time_step] = meal_info['cho_gpkg'] * patient_bw
            meal_announcements[time_step] = meal_info.get('announce', True)

    return meal_schedule, meal_announcements


def simulate_patient_scenario(
    digital_twin,
    controller,
    kalman_filter,
    meal_schedule: Dict[int, float],
    n_steps: int,
    controller_type: str = 'mpc',
    meal_announcements: Dict[int, bool] = None,
    collect_solver_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one scenario for a patient
    
    Args:
        digital_twin: Digital twin model
        controller: Controller instance
        kalman_filter: Kalman filter (if using MPC)
        meal_schedule: Meal schedule (actual CHO events)
        meal_announcements: Whether each meal is announced to the controller
        n_steps: Number of simulation steps
        controller_type: 'stochastic_mpc', 'deterministic_mpc', or 'pid'
    
    Returns:
        glucose_trace: Glucose measurements over time
        insulin_trace: Insulin infusion rates
        state_trace: Full state trajectory
    """
    glucose_trace = np.zeros(n_steps)
    insulin_trace = np.zeros(n_steps)
    state_trace = np.zeros((n_steps, digital_twin.n_states))
    solver_stats = {
        'solve_time_ms': [],
        'iterations': [],
        'status': [],
    }
    initial_sensor = digital_twin.get_measurement() + np.random.randn() * config.MEASUREMENT_NOISE_STD
    sensor_buffer = [initial_sensor] * (config.CGM_DELAY_STEPS + 1)
    
    for t in range(n_steps):
        try:
            # Get meal at this step
            meal_cho = meal_schedule.get(t, 0.0)
            current_glucose = sensor_buffer[0]
            
            # Determine control action
            if controller_type == 'pid':
                # PID uses only current glucose
                u = controller.solve(current_glucose)
            
            elif controller_type == 'deterministic_mpc':
                # Deterministic MPC uses state estimate
                if kalman_filter:
                    kalman_filter.update(current_glucose)
                    state_mean, _ = kalman_filter.get_state_estimate()
                    
                    # Build future meal schedule for prediction
                    future_meals = {}
                    for k, v in meal_schedule.items():
                        if t <= k < t + config.MPC_HORIZON:
                            if meal_announcements and not meal_announcements.get(k, True):
                                continue
                            future_meals[k - t] = v
                    
                    u = controller.solve(state_mean, future_meals)
                    kalman_filter.predict(u, meal_cho)
                else:
                    u = 0.0
            
            elif controller_type == 'stochastic_mpc':
                # Stochastic MPC uses state distribution
                if kalman_filter:
                    kalman_filter.update(current_glucose)
                    state_mean, state_cov = kalman_filter.get_state_estimate()
                    
                    # Build future meal schedule
                    future_meals = {}
                    for k, v in meal_schedule.items():
                        if t <= k < t + config.MPC_HORIZON:
                            if meal_announcements and not meal_announcements.get(k, True):
                                continue
                            future_meals[k - t] = v
                    
                    u, _, _, _ = controller.solve(state_mean, state_cov, future_meals)
                    kalman_filter.predict(u, meal_cho)
                else:
                    u = 0.0
            
            else:
                u = 0.0
            
            # Apply control and advance digital twin
            state, glucose = digital_twin.step(u, meal_cho, add_noise=True)
            sensor_buffer.append(glucose)
            sensor_buffer.pop(0)

            if collect_solver_stats:
                solver_stats['solve_time_ms'].append(float(getattr(controller, 'last_solve_time_ms', 0.0)))
                solver_stats['iterations'].append(int(getattr(controller, 'last_solve_iterations', 0)))
                solver_stats['status'].append(str(getattr(controller, 'last_solve_status', 'not_run')))
            
            # Record
            glucose_trace[t] = glucose
            insulin_trace[t] = u
            state_trace[t, :] = state
            
        except Exception as e:
            if t == 0: # Print error only once per patient to avoid spam
                print(f"Simulation error at step {t}: {e}")
                import traceback
                traceback.print_exc()
            
            # On error, use zero insulin and current glucose
            glucose_trace[t] = glucose_trace[t-1] if t > 0 else 120.0
            insulin_trace[t] = 0.0
            state_trace[t, :] = digital_twin.state
            if collect_solver_stats:
                solver_stats['solve_time_ms'].append(np.nan)
                solver_stats['iterations'].append(-1)
                solver_stats['status'].append('simulation_error')
    
    if collect_solver_stats:
        return glucose_trace, insulin_trace, state_trace, solver_stats
    return glucose_trace, insulin_trace, state_trace


def get_simulation_duration(scenario_type: str) -> int:
    """
    Get simulation duration in steps
    
    Args:
        scenario_type: Type of scenario
    
    Returns:
        Number of steps
    """
    sim_hours = config.SIMULATION_DAYS * 24 + config.WARMUP_HOURS
    n_steps = int(sim_hours * 60 / config.SAMPLING_TIME)
    return n_steps


def get_analysis_window(n_steps: int) -> Tuple[int, int]:
    """
    Get analysis window (exclude warmup)
    
    Args:
        n_steps: Total number of steps
    
    Returns:
        start_step, end_step
    """
    warmup_steps = int(config.WARMUP_HOURS * 60 / config.SAMPLING_TIME)
    return warmup_steps, n_steps
