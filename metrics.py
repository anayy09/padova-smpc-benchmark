"""
Glycemic metrics calculation (TIR, TBR, TAR, LBGI, HBGI, CV)
"""

import numpy as np
from typing import Dict, List
import config


def calculate_metrics(glucose_trace: np.ndarray) -> Dict[str, float]:
    """
    Calculate all glycemic metrics for a glucose trace
    
    Args:
        glucose_trace: Array of glucose values (mg/dL)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Time in range metrics
    metrics['TIR'] = calculate_TIR(glucose_trace)
    metrics['TBR'] = calculate_TBR(glucose_trace)
    metrics['TAR'] = calculate_TAR(glucose_trace)
    metrics['Time_below_54'] = calculate_severe_hypoglycemia(glucose_trace)
    
    # Summary statistics
    metrics['Mean_glucose'] = np.mean(glucose_trace)
    metrics['Std_glucose'] = np.std(glucose_trace)
    metrics['CV'] = calculate_CV(glucose_trace)
    
    # Risk indices
    metrics['LBGI'] = calculate_LBGI(glucose_trace)
    metrics['HBGI'] = calculate_HBGI(glucose_trace)
    
    # Min and max
    metrics['Min_glucose'] = np.min(glucose_trace)
    metrics['Max_glucose'] = np.max(glucose_trace)
    
    return metrics


def calculate_TIR(glucose: np.ndarray, lower: float = 70.0, upper: float = 180.0) -> float:
    """
    Calculate Time In Range (70-180 mg/dL)
    
    Args:
        glucose: Glucose trace
        lower: Lower bound (default 70 mg/dL)
        upper: Upper bound (default 180 mg/dL)
    
    Returns:
        TIR as percentage
    """
    in_range = (glucose >= lower) & (glucose <= upper)
    return 100.0 * np.mean(in_range)


def calculate_TBR(glucose: np.ndarray, threshold: float = 70.0) -> float:
    """
    Calculate Time Below Range (<70 mg/dL)
    
    Args:
        glucose: Glucose trace
        threshold: Hypoglycemia threshold (default 70 mg/dL)
    
    Returns:
        TBR as percentage
    """
    below = glucose < threshold
    return 100.0 * np.mean(below)


def calculate_TAR(glucose: np.ndarray, threshold: float = 180.0) -> float:
    """
    Calculate Time Above Range (>180 mg/dL)
    
    Args:
        glucose: Glucose trace
        threshold: Hyperglycemia threshold (default 180 mg/dL)
    
    Returns:
        TAR as percentage
    """
    above = glucose > threshold
    return 100.0 * np.mean(above)


def calculate_severe_hypoglycemia(glucose: np.ndarray, threshold: float = 54.0) -> float:
    """
    Calculate time below severe hypoglycemia threshold
    
    Args:
        glucose: Glucose trace
        threshold: Severe hypo threshold (default 54 mg/dL)
    
    Returns:
        Time below threshold as percentage
    """
    severe_hypo = glucose < threshold
    return 100.0 * np.mean(severe_hypo)


def calculate_CV(glucose: np.ndarray) -> float:
    """
    Calculate Coefficient of Variation
    
    Args:
        glucose: Glucose trace
    
    Returns:
        CV as percentage
    """
    mean_g = np.mean(glucose)
    std_g = np.std(glucose)
    if mean_g > 0:
        return 100.0 * std_g / mean_g
    else:
        return 0.0


def calculate_LBGI(glucose: np.ndarray) -> float:
    """
    Calculate Low Blood Glucose Index
    
    Args:
        glucose: Glucose trace (mg/dL)
    
    Returns:
        LBGI value
    """
    glucose_safe = np.clip(glucose, 1.0, None)
    f = 1.509 * (np.log(glucose_safe)**1.084 - 5.381)
    
    # Risk function for low glucose
    rl = 10 * f**2
    rl[glucose >= 112.5] = 0  # No risk above 112.5 mg/dL
    
    # LBGI is mean of risk values
    return np.mean(rl)


def calculate_HBGI(glucose: np.ndarray) -> float:
    """
    Calculate High Blood Glucose Index
    
    Args:
        glucose: Glucose trace (mg/dL)
    
    Returns:
        HBGI value
    """
    glucose_safe = np.clip(glucose, 1.0, None)
    f = 1.509 * (np.log(glucose_safe)**1.084 - 5.381)
    
    # Risk function for high glucose
    rh = 10 * f**2
    rh[glucose <= 112.5] = 0  # No risk below 112.5 mg/dL
    
    # HBGI is mean of risk values
    return np.mean(rh)


def print_metrics(metrics: Dict[str, float], controller_name: str = ""):
    """
    Print metrics in formatted table
    
    Args:
        metrics: Dictionary of metrics
        controller_name: Name of controller
    """
    if controller_name:
        print(f"\n{'='*60}")
        print(f"Metrics for {controller_name}")
        print(f"{'='*60}")
    
    print(f"Time in Range (70-180 mg/dL):     {metrics['TIR']:.1f}%")
    print(f"Time Below Range (<70 mg/dL):     {metrics['TBR']:.1f}%")
    print(f"Time Above Range (>180 mg/dL):    {metrics['TAR']:.1f}%")
    print(f"Time Below 54 mg/dL:              {metrics['Time_below_54']:.1f}%")
    print(f"\nMean Glucose:                     {metrics['Mean_glucose']:.1f} mg/dL")
    print(f"Standard Deviation:               {metrics['Std_glucose']:.1f} mg/dL")
    print(f"Coefficient of Variation:         {metrics['CV']:.1f}%")
    print(f"\nLow Blood Glucose Index (LBGI):   {metrics['LBGI']:.2f}")
    print(f"High Blood Glucose Index (HBGI):  {metrics['HBGI']:.2f}")
    print(f"\nMinimum Glucose:                  {metrics['Min_glucose']:.1f} mg/dL")
    print(f"Maximum Glucose:                  {metrics['Max_glucose']:.1f} mg/dL")


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple patients
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Dictionary with mean and std for each metric
    """
    aggregated = {}
    
    if not metrics_list:
        return aggregated
    
    # Get all metric keys
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return aggregated


def print_aggregated_metrics(aggregated: Dict[str, Dict[str, float]], controller_name: str = ""):
    """
    Print aggregated metrics
    
    Args:
        aggregated: Aggregated metrics dictionary
        controller_name: Name of controller
    """
    if controller_name:
        print(f"\n{'='*70}")
        print(f"Aggregated Metrics for {controller_name}")
        print(f"{'='*70}")
    
    print(f"\n{'Metric':<35} {'Mean ± Std':<20} {'Median':<10}")
    print(f"{'-'*70}")
    
    metrics_order = ['TIR', 'TBR', 'TAR', 'Time_below_54', 'Mean_glucose', 
                     'Std_glucose', 'CV', 'LBGI', 'HBGI']
    
    for key in metrics_order:
        if key in aggregated:
            mean_val = aggregated[key]['mean']
            std_val = aggregated[key]['std']
            median_val = aggregated[key]['median']
            
            if 'glucose' in key.lower() and key != 'CV':
                print(f"{key:<35} {mean_val:6.1f} ± {std_val:5.1f} mg/dL  {median_val:6.1f}")
            elif 'Time' in key or 'TIR' in key or 'TBR' in key or 'TAR' in key or 'CV' in key:
                print(f"{key:<35} {mean_val:6.1f} ± {std_val:5.1f} %      {median_val:6.1f}")
            else:
                print(f"{key:<35} {mean_val:6.2f} ± {std_val:5.2f}        {median_val:6.2f}")
