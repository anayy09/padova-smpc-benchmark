from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from run_experiments import run_single_experiment


OUTPUT_PATH = Path(config.DATA_DIR) / "runtime_profile_summary.csv"


def summarize_solver_stats(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(finite)), float(np.percentile(finite, 95)), float(np.max(finite))


def profile_controller(controller: str, scenario: str, cohort: str, num_patients: int) -> dict[str, object]:
    wall_times = []
    solve_times = []
    iterations = []
    statuses = []
    successful_runs = 0

    for patient_id in range(num_patients):
        start = time.perf_counter()
        result = run_single_experiment(
            patient_id=patient_id,
            scenario_type=scenario,
            controller_type=controller,
            cohort=cohort,
            collect_solver_stats=True,
        )
        wall_times.append((time.perf_counter() - start) * 1000.0)
        solver_stats = result.get("solver_stats", {})
        solve_times.extend(solver_stats.get("solve_time_ms", []))
        iterations.extend(solver_stats.get("iterations", []))
        statuses.extend(solver_stats.get("status", []))
        successful_runs += 1

    mean_solve_ms, p95_solve_ms, max_solve_ms = summarize_solver_stats(solve_times)
    wall_array = np.asarray(wall_times, dtype=float)
    valid_iterations = np.asarray([value for value in iterations if value >= 0], dtype=float)
    status_series = pd.Series(statuses, dtype="string")

    return {
        "controller": controller,
        "scenario": scenario,
        "cohort": cohort,
        "patients_profiled": successful_runs,
        "control_updates": int(len(solve_times)),
        "mean_solve_time_ms": mean_solve_ms,
        "p95_solve_time_ms": p95_solve_ms,
        "max_solve_time_ms": max_solve_ms,
        "mean_iterations": float(valid_iterations.mean()) if valid_iterations.size else float("nan"),
        "p95_iterations": float(np.percentile(valid_iterations, 95)) if valid_iterations.size else float("nan"),
        "solver_success_rate": float((status_series == "solved").mean()) if len(status_series) else float("nan"),
        "mean_wall_time_per_subject_ms": float(wall_array.mean()) if wall_array.size else float("nan"),
    }


def main() -> None:
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    num_patients = min(20, config.NUM_PATIENTS)
    rows = []

    for controller in ["stochastic_mpc", "deterministic_mpc", "pid"]:
        for scenario in config.MEAL_SCENARIOS:
            for cohort in config.COHORTS:
                print(f"Profiling {controller} | {scenario} | {cohort} | n={num_patients}")
                rows.append(profile_controller(controller, scenario, cohort, num_patients))

    summary = pd.DataFrame(rows)
    controller_summary = (
        summary.groupby("controller", as_index=False)
        .agg(
            profiled_cells=("controller", "size"),
            patients_profiled=("patients_profiled", "sum"),
            control_updates=("control_updates", "sum"),
            mean_solve_time_ms=("mean_solve_time_ms", "mean"),
            p95_solve_time_ms=("p95_solve_time_ms", "mean"),
            max_solve_time_ms=("max_solve_time_ms", "max"),
            mean_iterations=("mean_iterations", "mean"),
            p95_iterations=("p95_iterations", "mean"),
            solver_success_rate=("solver_success_rate", "mean"),
            mean_wall_time_per_subject_ms=("mean_wall_time_per_subject_ms", "mean"),
        )
    )

    summary.to_csv(OUTPUT_PATH, index=False)
    controller_summary.to_csv(OUTPUT_PATH.with_name("runtime_profile_by_controller.csv"), index=False)

    excel_path = OUTPUT_PATH.with_suffix(".xlsx")
    try:
        with pd.ExcelWriter(excel_path) as writer:
            summary.to_excel(writer, sheet_name="by_cell", index=False)
            controller_summary.to_excel(writer, sheet_name="by_controller", index=False)
        excel_status = str(excel_path)
    except ModuleNotFoundError:
        excel_status = "skipped (openpyxl not installed)"

    print("Saved runtime profiling outputs:")
    print(f" - {OUTPUT_PATH}")
    print(f" - {OUTPUT_PATH.with_name('runtime_profile_by_controller.csv')}")
    print(f" - {excel_status}")
    print("\nController summary:")
    print(controller_summary.to_string(index=False))


if __name__ == "__main__":
    main()