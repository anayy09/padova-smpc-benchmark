"""
Hyperparameter grid search restricted to the designated training partition.

Runs SMPC, DMPC, and PID on TRAIN_SUBJECT_IDS (first 30 adult subjects, seed 42)
under the standard scenario and reports the best cost-function weights.
Results are written to results/data/hyperparameter_tuning.json.

Usage:
    python tune_hyperparameters.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

import config
from run_experiments import run_single_experiment
from train_test_split import TRAIN_SUBJECT_IDS, TUNING_COHORT, TUNING_SCENARIO

SCENARIO = TUNING_SCENARIO   # 'standard'
COHORT   = TUNING_COHORT     # 'adult'
OUTPUT   = Path(config.DATA_DIR) / "hyperparameter_tuning.json"

# ------------------------------------------------------------------
# Grid definitions — coarse sweep; refine manually if needed
# ------------------------------------------------------------------
SMPC_GRIDS = {
    "W_GLUCOSE":    [0.5, 1.0, 2.0],
    "W_VARIANCE":   [0.25, 0.5, 1.0],
    "W_INSULIN":    [0.005, 0.01, 0.02],
    "W_LOW_GLUCOSE":[3.0, 5.0, 8.0],
}

DMPC_GRIDS = {
    "W_GLUCOSE":  [0.5, 1.0, 2.0],
    "W_INSULIN":  [0.005, 0.01, 0.02],
    "W_DELTA_U":  [0.05, 0.1, 0.2],
}


def _score(metrics: dict) -> float:
    """Lower is better: penalise TBR heavily, reward TIR."""
    return metrics["TBR"] * 5.0 - metrics["TIR"] + metrics["LBGI"] * 2.0


def _run_grid(controller_type: str, grid: dict[str, list]) -> dict:
    """Exhaustive grid search over TRAIN_SUBJECT_IDS."""
    keys   = list(grid.keys())
    values = list(grid.values())
    best_score   = float("inf")
    best_params: dict = {}
    best_metrics: dict = {}

    total = 1
    for v in values:
        total *= len(v)
    print(f"\n  {controller_type}: {total} combinations × {len(TRAIN_SUBJECT_IDS)} subjects")

    for combo in product(*values):
        params = dict(zip(keys, combo))
        # Temporarily override config
        original = {k: getattr(config, k) for k in params}
        for k, v in params.items():
            setattr(config, k, v)

        scores = []
        for pid in TRAIN_SUBJECT_IDS:
            try:
                result  = run_single_experiment(pid, SCENARIO, controller_type, COHORT)
                scores.append(_score(result["metrics"]))
            except Exception:
                scores.append(1e6)

        # Restore config
        for k, v in original.items():
            setattr(config, k, v)

        mean_score = float(np.mean(scores))
        if mean_score < best_score:
            best_score   = mean_score
            best_params  = params.copy()

    # Re-run with best params to get readable metrics
    for k, v in best_params.items():
        setattr(config, k, v)
    all_m = []
    for pid in TRAIN_SUBJECT_IDS:
        try:
            r = run_single_experiment(pid, SCENARIO, controller_type, COHORT)
            all_m.append(r["metrics"])
        except Exception:
            pass
    for k, v in {k: getattr(config, k) for k in best_params}.items():
        setattr(config, k, getattr(config, k))  # no-op restore (already set)

    if all_m:
        best_metrics = {
            "TIR_mean":  float(np.mean([m["TIR"]  for m in all_m])),
            "TBR_mean":  float(np.mean([m["TBR"]  for m in all_m])),
            "LBGI_mean": float(np.mean([m["LBGI"] for m in all_m])),
        }

    print(f"  Best params: {best_params}  →  {best_metrics}")
    return {"params": best_params, "train_metrics": best_metrics}


def main() -> None:
    np.random.seed(42)
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Hyperparameter tuning — training partition only")
    print(f"Cohort: {COHORT}, Scenario: {SCENARIO}, Subjects: {TRAIN_SUBJECT_IDS}")
    print("=" * 70)

    results = {
        "cohort":   COHORT,
        "scenario": SCENARIO,
        "train_subject_ids": TRAIN_SUBJECT_IDS,
        "current_config": {
            "W_GLUCOSE":    config.W_GLUCOSE,
            "W_VARIANCE":   config.W_VARIANCE,
            "W_INSULIN":    config.W_INSULIN,
            "W_DELTA_U":    config.W_DELTA_U,
            "W_LOW_GLUCOSE":config.W_LOW_GLUCOSE,
        },
    }

    results["smpc"] = _run_grid("stochastic_mpc",  SMPC_GRIDS)
    results["dmpc"] = _run_grid("deterministic_mpc", DMPC_GRIDS)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nResults written to {OUTPUT}")
    print("\nCompare best params against current config.py values.")
    print("If they differ by more than 20%, update config.py and re-run experiments.")


if __name__ == "__main__":
    main()
