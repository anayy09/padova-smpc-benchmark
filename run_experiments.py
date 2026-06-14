"""
Main experimental script for running in-silico trials
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Dict, Tuple

import config
from digital_twin import StochasticDigitalTwin
from kalman_filter import ExtendedKalmanFilter
from controllers_improved import StochasticMPC, DeterministicMPC, PIDController
from metrics import calculate_metrics, aggregate_metrics, print_aggregated_metrics
from scenarios import (generate_patient_parameters, generate_meal_scenario, 
                      simulate_patient_scenario, get_simulation_duration, 
                      get_analysis_window)


def setup_directories():
    """Create output directories"""
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    Path(config.FIGURES_DIR).mkdir(exist_ok=True)
    Path(config.DATA_DIR).mkdir(exist_ok=True)


def run_single_experiment(patient_id: int, scenario_type: str, controller_type: str, cohort: str, collect_solver_stats: bool = False):
    """
    Run single experiment for one patient, scenario, and controller
    
    Args:
        patient_id: Patient identifier
        scenario_type: 'standard', 'variable', or 'nocturnal'
        controller_type: 'stochastic_mpc', 'deterministic_mpc', or 'pid'
        cohort: Subject cohort ('adult', 'adolescent', 'child')
    
    Returns:
        results: Dictionary with glucose trace, insulin trace, and metrics
    """
    # Generate patient parameters
    patient_params = generate_patient_parameters(patient_id, cohort)
    
    # Create digital twin
    digital_twin = StochasticDigitalTwin(patient_params, dt=config.SAMPLING_TIME)
    digital_twin.reset(initial_glucose=120.0)
    
    # Create controller
    kalman_filter = None
    if controller_type == 'stochastic_mpc':
        controller = StochasticMPC(digital_twin)
        kalman_filter = ExtendedKalmanFilter(digital_twin)
    elif controller_type == 'deterministic_mpc':
        controller = DeterministicMPC(digital_twin)
        kalman_filter = ExtendedKalmanFilter(digital_twin)
    else:  # PID
        controller = PIDController(cohort=cohort)
    
    if hasattr(controller, 'reset'):
        controller.reset()

    # Generate body-weight-normalised meal scenario
    patient_bw = patient_params.get('BW', 75.0)
    meal_schedule, meal_announcements = generate_meal_scenario(
        scenario_type, day=patient_id, patient_bw=patient_bw
    )
    
    # Get simulation duration
    n_steps = get_simulation_duration(scenario_type)
    
    # Run simulation
    simulation_output = simulate_patient_scenario(
        digital_twin,
        controller,
        kalman_filter,
        meal_schedule,
        n_steps,
        controller_type,
        meal_announcements=meal_announcements,
        collect_solver_stats=collect_solver_stats,
    )
    if collect_solver_stats:
        glucose_trace, insulin_trace, state_trace, solver_stats = simulation_output
    else:
        glucose_trace, insulin_trace, state_trace = simulation_output
    
    # Extract analysis window (exclude warmup)
    start_step, end_step = get_analysis_window(n_steps)
    glucose_analysis = glucose_trace[start_step:end_step]
    insulin_analysis = insulin_trace[start_step:end_step]
    
    # Calculate metrics
    metrics = calculate_metrics(glucose_analysis)
    metrics['Total_insulin'] = np.sum(insulin_analysis) * config.SAMPLING_TIME / 60.0  # Total U
    
    results = {
        'patient_id': patient_id,
        'scenario': scenario_type,
        'controller': controller_type,
        'cohort': cohort,
        'glucose': glucose_trace,
        'insulin': insulin_trace,
        'metrics': metrics,
        'meal_schedule': meal_schedule,
        'meal_announcements': meal_announcements,
    }
    if collect_solver_stats:
        results['solver_stats'] = solver_stats
    
    return results


def run_cohort_experiment(scenario_type: str, controller_type: str, cohort: str, num_patients: int = None):
    """
    Run experiment across patient cohort
    
    Args:
        scenario_type: Scenario type
        controller_type: Controller type
        num_patients: Number of patients (default: from config)
    
    Returns:
        results_list: List of result dictionaries
    """
    if num_patients is None:
        num_patients = config.NUM_PATIENTS
    
    results_list = []
    
    print(f"\nRunning {controller_type} on {scenario_type} scenario ({cohort}) for {num_patients} patients...")
    
    for patient_id in tqdm(range(num_patients)):
        try:
            results = run_single_experiment(patient_id, scenario_type, controller_type, cohort)
            results_list.append(results)
        except Exception as e:
            print(f"\nError for patient {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results_list


def compare_controllers(scenario_type: str = 'standard', cohort: str = 'adult', num_patients: int = None):
    """
    Compare all three controllers on a given scenario
    
    Args:
        scenario_type: Scenario type
        num_patients: Number of patients
    
    Returns:
        comparison_results: Dictionary of results for each controller
    """
    if num_patients is None:
        num_patients = config.NUM_PATIENTS
    
    controllers = ['stochastic_mpc', 'deterministic_mpc', 'pid']
    comparison_results = {}
    
    for controller_type in controllers:
        results_list = run_cohort_experiment(scenario_type, controller_type, cohort, num_patients)
        comparison_results[controller_type] = results_list
    
    return comparison_results


def save_results(comparison_results: Dict, scenario_type: str, cohort: str):
    """Save results to disk"""
    filename = f"{config.DATA_DIR}/results_{scenario_type}_{cohort}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(comparison_results, f)
    print(f"\nResults saved to {filename}")


def load_results(scenario_type: str, cohort: str):
    """Load results from disk"""
    filename = f"{config.DATA_DIR}/results_{scenario_type}_{cohort}.pkl"
    with open(filename, 'rb') as f:
        comparison_results = pickle.load(f)
    return comparison_results


def plot_glucose_traces(comparison_results: Dict, scenario_type: str, cohort: str, patient_id: int = 0):
    """
    Plot glucose traces for all controllers for one patient
    
    Args:
        comparison_results: Results dictionary
        scenario_type: Scenario type
        patient_id: Which patient to plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    controllers = ['stochastic_mpc', 'deterministic_mpc', 'pid']
    titles = ['Stochastic MPC', 'Deterministic MPC', 'PID']
    
    time_hours = np.arange(0, len(comparison_results['pid'][patient_id]['glucose'])) * config.SAMPLING_TIME / 60.0
    
    for idx, (controller, title) in enumerate(zip(controllers, titles)):
        ax = axes[idx]
        
        results = comparison_results[controller][patient_id]
        glucose = results['glucose']
        
        ax.plot(time_hours, glucose, label='Glucose', linewidth=1.5)
        ax.axhline(config.TIR_LOWER, color='green', linestyle='--', alpha=0.5, label='TIR bounds')
        ax.axhline(config.TIR_UPPER, color='green', linestyle='--', alpha=0.5)
        ax.axhline(config.G_SEVERE_HYPO, color='red', linestyle='--', alpha=0.5, label='Severe hypo')
        
        # Mark meals
        for time_step, cho in results['meal_schedule'].items():
            meal_hour = time_step * config.SAMPLING_TIME / 60.0
            ax.axvline(meal_hour, color='orange', linestyle=':', alpha=0.3)
        
        ax.set_ylabel('Glucose (mg/dL)', fontsize=11)
        ax.set_title(f'{title} - Patient {patient_id}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 400)
        
        if idx == 2:
            ax.set_xlabel('Time (hours)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(
        f"{config.FIGURES_DIR}/glucose_comparison_{scenario_type}_{cohort}_patient{patient_id}.png",
        dpi=300,
    )
    plt.close()


def create_metrics_table(comparison_results: Dict, scenario_type: str, cohort: str):
    """
    Create table comparing metrics across controllers
    
    Args:
        comparison_results: Results dictionary
        scenario_type: Scenario type
    
    Returns:
        DataFrame with comparison
    """
    data = []
    
    for controller_type, results_list in comparison_results.items():
        metrics_list = [r['metrics'] for r in results_list]
        aggregated = aggregate_metrics(metrics_list)
        
        row = {
            'Controller': controller_type.replace('_', ' ').title(),
            'TIR (%)': f"{aggregated['TIR']['mean']:.1f} ± {aggregated['TIR']['std']:.1f}",
            'TBR (%)': f"{aggregated['TBR']['mean']:.1f} ± {aggregated['TBR']['std']:.1f}",
            'TAR (%)': f"{aggregated['TAR']['mean']:.1f} ± {aggregated['TAR']['std']:.1f}",
            'Time <54 (%)': f"{aggregated['Time_below_54']['mean']:.1f} ± {aggregated['Time_below_54']['std']:.1f}",
            'Mean Glucose (mg/dL)': f"{aggregated['Mean_glucose']['mean']:.1f} ± {aggregated['Mean_glucose']['std']:.1f}",
            'CV (%)': f"{aggregated['CV']['mean']:.1f} ± {aggregated['CV']['std']:.1f}",
            'LBGI': f"{aggregated['LBGI']['mean']:.2f} ± {aggregated['LBGI']['std']:.2f}",
            'HBGI': f"{aggregated['HBGI']['mean']:.2f} ± {aggregated['HBGI']['std']:.2f}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(f"{config.DATA_DIR}/metrics_comparison_{scenario_type}_{cohort}.csv", index=False)
    
    # Print
    print(f"\n{'='*100}")
    print(f"Metrics Comparison - {scenario_type.title()} Scenario ({cohort.title()})")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    
    return df


def plot_metrics_comparison(comparison_results: Dict, scenario_type: str, cohort: str):
    """
    Create bar plots comparing key metrics
    
    Args:
        comparison_results: Results dictionary
        scenario_type: Scenario type
    """
    metrics_to_plot = ['TIR', 'TBR', 'Time_below_54', 'CV', 'LBGI', 'HBGI']
    titles = ['Time in Range (%)', 'Time Below Range (%)', 
              'Time Below 54 mg/dL (%)', 'Coefficient of Variation (%)',
              'Low Blood Glucose Index', 'High Blood Glucose Index']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    controllers = list(comparison_results.keys())
    controller_labels = [c.replace('_', ' ').title() for c in controllers]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        
        means = []
        stds = []
        
        for controller_type in controllers:
            results_list = comparison_results[controller_type]
            metrics_list = [r['metrics'] for r in results_list]
            aggregated = aggregate_metrics(metrics_list)
            
            means.append(aggregated[metric]['mean'])
            stds.append(aggregated[metric]['std'])
        
        x = np.arange(len(controllers))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(controller_labels, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(
        f'Controller Comparison - {scenario_type.title()} Scenario ({cohort.title()})',
        fontsize=14,
        fontweight='bold',
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(
        f"{config.FIGURES_DIR}/metrics_comparison_{scenario_type}_{cohort}.png",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def main():
    """Main experimental pipeline"""
    print("="*80)
    print("Digital Twin Stochastic MPC - In Silico Experiments")
    print("="*80)
    
    # Setup
    setup_directories()
    np.random.seed(42)
    
    # Run experiments for each scenario type
    scenarios = ['standard', 'variable', 'nocturnal']
    
    for scenario_type in scenarios:
        for cohort in config.COHORTS:
            print(f"\n\n{'='*80}")
            print(f"Running experiments for {scenario_type.upper()} scenario ({cohort.upper()})")
            print(f"{'='*80}")
            
            comparison_results = compare_controllers(
                scenario_type, cohort, num_patients=config.NUM_PATIENTS
            )
            save_results(comparison_results, scenario_type, cohort)
            create_metrics_table(comparison_results, scenario_type, cohort)
            plot_metrics_comparison(comparison_results, scenario_type, cohort)
            
            for patient_id in range(min(3, config.NUM_PATIENTS)):
                plot_glucose_traces(comparison_results, scenario_type, cohort, patient_id)
            
            print(f"\nCompleted {scenario_type} scenario ({cohort})")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
