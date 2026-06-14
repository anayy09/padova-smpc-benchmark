"""Generate manuscript-ready statistical summaries across all scenarios and cohorts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config

DATA_DIR = Path(config.DATA_DIR)
RESULT_PATTERN = "results_{scenario}_{cohort}.pkl"
OUTPUT_CSV = DATA_DIR / "statistical_comparison_summary.csv"
PRIMARY_OUTPUT_CSV = DATA_DIR / "primary_endpoint_summary.csv"


def load_results(scenario: str, cohort: str) -> Dict[str, List[Dict]]:
    path = DATA_DIR / RESULT_PATTERN.format(scenario=scenario, cohort=cohort)
    with path.open("rb") as handle:
        return pickle.load(handle)


def bootstrap_ci(differences: np.ndarray, *, n_boot: int = 4000, alpha: float = 0.05) -> tuple[float, float]:
    if len(differences) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for idx in range(n_boot):
        sample = rng.choice(differences, size=len(differences), replace=True)
        boots[idx] = np.mean(sample)
    lower = np.quantile(boots, alpha / 2)
    upper = np.quantile(boots, 1 - alpha / 2)
    return float(lower), float(upper)


def rank_biserial_from_wilcoxon(statistic: float, n_obs: int) -> float:
    if n_obs == 0:
        return float("nan")
    total_rank = n_obs * (n_obs + 1) / 2
    return float(1 - 2 * statistic / total_rank)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    ranked = p_values.rank(method="first").astype(int)
    adjusted = p_values * len(p_values) / ranked
    adjusted = adjusted.clip(upper=1.0)
    adjusted = adjusted.sort_values(ascending=False).cummin().sort_index()
    return adjusted


def build_summary() -> pd.DataFrame:
    rows = []
    metrics = ["TIR", "TBR", "Time_below_54", "Mean_glucose", "CV", "LBGI", "HBGI"]

    for scenario in ("standard", "variable", "nocturnal"):
        for cohort in config.COHORTS:
            comparison_results = load_results(scenario, cohort)
            smpc_runs = {run["patient_id"]: run for run in comparison_results["stochastic_mpc"]}
            dmpc_runs = {run["patient_id"]: run for run in comparison_results["deterministic_mpc"]}

            common_ids = sorted(set(smpc_runs) & set(dmpc_runs))
            for metric in metrics:
                smpc = np.array([smpc_runs[idx]["metrics"][metric] for idx in common_ids], dtype=float)
                dmpc = np.array([dmpc_runs[idx]["metrics"][metric] for idx in common_ids], dtype=float)
                diffs = smpc - dmpc
                try:
                    stat, p_value = wilcoxon(smpc, dmpc, zero_method="pratt", alternative="two-sided")
                except ValueError:
                    stat, p_value = np.nan, np.nan
                ci_low, ci_high = bootstrap_ci(diffs)
                rows.append(
                    {
                        "scenario": scenario,
                        "cohort": cohort,
                        "metric": metric,
                        "n": len(common_ids),
                        "smpc_mean": float(np.mean(smpc)),
                        "dmpc_mean": float(np.mean(dmpc)),
                        "mean_difference": float(np.mean(diffs)),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "median_difference": float(np.median(diffs)),
                        "wilcoxon_stat": float(stat) if not np.isnan(stat) else np.nan,
                        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                        "rank_biserial": rank_biserial_from_wilcoxon(stat, len(common_ids)) if not np.isnan(stat) else np.nan,
                    }
                )

    summary = pd.DataFrame(rows)
    summary["q_value_bh"] = benjamini_hochberg(summary["p_value"].fillna(1.0))
    summary = summary.sort_values(["metric", "scenario", "cohort"]).reset_index(drop=True)
    return summary


def build_primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[summary["metric"].isin(["TIR", "TBR", "Time_below_54", "LBGI"])].copy()
    primary["mean_diff_text"] = primary.apply(
        lambda row: f"{row['mean_difference']:.2f} [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]",
        axis=1,
    )
    primary["p_text"] = primary["p_value"].map(lambda value: f"{value:.3g}")
    primary["q_text"] = primary["q_value_bh"].map(lambda value: f"{value:.3g}")
    return primary[
        [
            "scenario",
            "cohort",
            "metric",
            "smpc_mean",
            "dmpc_mean",
            "mean_diff_text",
            "rank_biserial",
            "p_text",
            "q_text",
        ]
    ]


def main() -> None:
    summary = build_summary()
    summary.to_csv(OUTPUT_CSV, index=False)
    primary = build_primary_table(summary)
    primary.to_csv(PRIMARY_OUTPUT_CSV, index=False)
    print(f"Saved full statistics to {OUTPUT_CSV}")
    print(f"Saved primary endpoint summary to {PRIMARY_OUTPUT_CSV}")


if __name__ == "__main__":
    main()