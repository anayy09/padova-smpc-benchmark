from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import config
from run_experiments import run_single_experiment


OUTPUT_PATH = Path(config.DATA_DIR) / "targeted_ablation_child_disturbance.csv"


@contextmanager
def temporary_config(overrides: dict[str, object]):
    original = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


def run_variant(variant_name: str, scenario: str, patient_id: int, overrides: dict[str, object], controller_type: str) -> dict[str, object]:
    with temporary_config(overrides):
        result = run_single_experiment(
            patient_id=patient_id,
            scenario_type=scenario,
            controller_type=controller_type,
            cohort="child",
        )
    metrics = result["metrics"]
    return {
        "variant": variant_name,
        "scenario": scenario,
        "patient_id": patient_id,
        "controller_type": controller_type,
        "TIR": metrics["TIR"],
        "TBR": metrics["TBR"],
        "Time_below_54": metrics["Time_below_54"],
        "Mean_glucose": metrics["Mean_glucose"],
        "CV": metrics["CV"],
        "LBGI": metrics["LBGI"],
    }


def main() -> None:
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    scenarios = ["variable", "nocturnal"]
    patient_ids = range(min(20, config.NUM_PATIENTS))
    variants = [
        ("SMPC full", "stochastic_mpc", {}),
        ("SMPC no variance penalty", "stochastic_mpc", {"W_VARIANCE": 0.0}),
        ("SMPC no chance tightening", "stochastic_mpc", {"KAPPA_HYPO": 0.0, "KAPPA_HYPER": 0.0, "W_RISK_SLACK": 0.0, "W_LOW_GLUCOSE": 0.0}),
        ("SMPC short horizon", "stochastic_mpc", {"MPC_HORIZON": 6}),
        ("DMPC baseline", "deterministic_mpc", {}),
    ]

    rows = []
    for scenario in scenarios:
        for variant_name, controller_type, overrides in variants:
            print(f"Running {variant_name} | {scenario} | n={len(list(patient_ids))}")
            for patient_id in patient_ids:
                rows.append(run_variant(variant_name, scenario, patient_id, overrides, controller_type))

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["scenario", "variant"], as_index=False)
        .agg(
            subjects=("patient_id", "count"),
            TIR_mean=("TIR", "mean"),
            TBR_mean=("TBR", "mean"),
            Time_below_54_mean=("Time_below_54", "mean"),
            Mean_glucose_mean=("Mean_glucose", "mean"),
            CV_mean=("CV", "mean"),
            LBGI_mean=("LBGI", "mean"),
        )
    )

    df.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(OUTPUT_PATH.with_name("targeted_ablation_child_disturbance_summary.csv"), index=False)

    print("Saved ablation outputs:")
    print(f" - {OUTPUT_PATH}")
    print(f" - {OUTPUT_PATH.with_name('targeted_ablation_child_disturbance_summary.csv')}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()