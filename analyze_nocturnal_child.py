"""Post-processing script for nocturnal child cohort visualizations and stats."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

import config

DATA_FILE = Path(config.DATA_DIR) / "results_nocturnal_child.pkl"
FIG_GLUCOSE = Path(config.FIGURES_DIR) / "nocturnal_child_glucose.png"
FIG_CVGA = Path(config.FIGURES_DIR) / "nocturnal_child_cvga.png"
STATS_CSV = Path(config.DATA_DIR) / "nocturnal_child_wilcoxon.csv"

sns.set_context("talk")
sns.set_style("whitegrid")


def load_results() -> Dict[str, List[Dict]]:
    with DATA_FILE.open("rb") as f:
        data = pickle.load(f)
    return data


def results_to_dataframe(comparison_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    rows = []
    for controller, runs in comparison_results.items():
        cohort_rows = []
        for run in runs:
            metrics = run["metrics"].copy()
            metrics["patient_id"] = run["patient_id"]
            metrics["controller"] = controller
            cohort_rows.append(metrics)
        rows.extend(cohort_rows)
    df = pd.DataFrame(rows)
    return df


def choose_representative_patient(df: pd.DataFrame) -> int:
    patient_ids = np.sort(df["patient_id"].unique())
    rng = np.random.default_rng(42)
    return int(rng.choice(patient_ids))


def plot_glucose_traces(comparison_results: Dict[str, List[Dict]], patient_id: int) -> None:
    time_hours = None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"stochastic_mpc": "#2E86AB", "deterministic_mpc": "#A23B72"}
    labels = {
        "stochastic_mpc": "Stochastic MPC",
        "deterministic_mpc": "Deterministic MPC",
    }
    for controller in ("stochastic_mpc", "deterministic_mpc"):
        patient_runs = [r for r in comparison_results[controller] if r["patient_id"] == patient_id]
        if not patient_runs:
            continue
        run = patient_runs[0]
        glucose = np.array(run["glucose"])
        if time_hours is None:
            time_hours = np.arange(len(glucose)) * config.SAMPLING_TIME / 60.0
        ax.plot(time_hours, glucose, label=labels[controller], color=colors[controller], linewidth=1.7)
    ax.axhspan(config.TIR_LOWER, config.TIR_UPPER, color="#E8F6EF", alpha=0.5, label="Target 70-180 mg/dL")
    ax.axhline(config.G_SEVERE_HYPO, color="#C0392B", linestyle="--", linewidth=1.2, label="54 mg/dL")
    ax.set_xlim(time_hours[0], time_hours[-1])
    ax.set_ylim(40, 260)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"Fixed Random Nocturnal Child Scenario - Patient {patient_id}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_GLUCOSE, dpi=300)
    plt.close(fig)


def plot_cvga(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    limits_mean = (70, 190)
    limits_std = (0, 120)
    target_rect = {
        "xmin": 90,
        "xmax": 140,
        "ymin": 0,
        "ymax": 45,
    }
    for ax, controller, title in zip(
        axes,
        ("stochastic_mpc", "deterministic_mpc"),
        ("Stochastic MPC", "Deterministic MPC"),
    ):
        sub = df[df["controller"] == controller]
        ax.scatter(
            sub["Mean_glucose"],
            sub["Std_glucose"],
            c="#2E86AB" if controller == "stochastic_mpc" else "#A23B72",
            alpha=0.65,
            edgecolor="black",
            linewidth=0.4,
            s=45,
        )
        ax.axvline(config.TIR_LOWER, color="#C0392B", linestyle="-", linewidth=2.0, alpha=0.9, label="70 mg/dL")
        ax.text(config.TIR_LOWER + 1, limits_std[1] * 0.92, "70 mg/dL", color="#C0392B", fontsize=9, va="top")
        ax.axvline(config.TIR_UPPER, color="#7B7D7D", linestyle="--", linewidth=1.5)
        ax.axhline(50, color="#7B7D7D", linestyle=":", linewidth=1)
        rect = Rectangle(
            (target_rect["xmin"], target_rect["ymin"]),
            target_rect["xmax"] - target_rect["xmin"],
            target_rect["ymax"] - target_rect["ymin"],
            color="#ABEBC6",
            alpha=0.3,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.set_title(title)
        ax.set_xlabel("Mean glucose (mg/dL)")
    axes[0].set_ylabel("Std. dev (mg/dL)")
    axes[0].set_xlim(*limits_mean)
    axes[0].set_ylim(*limits_std)
    fig.suptitle("Control Variability Grid - Nocturnal Child Cohort", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_CVGA, dpi=300)
    plt.close(fig)


def bootstrap_ci(differences: np.ndarray, *, n_boot: int = 4000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for idx in range(n_boot):
        sample = rng.choice(differences, size=len(differences), replace=True)
        boots[idx] = np.mean(sample)
    lower = np.quantile(boots, alpha / 2)
    upper = np.quantile(boots, 1 - alpha / 2)
    return float(lower), float(upper)


def rank_biserial_from_wilcoxon(statistic: float, n_obs: int) -> float:
    total_rank = n_obs * (n_obs + 1) / 2
    return float(1 - 2 * statistic / total_rank)


def run_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {}
    pivot = df.pivot(index="patient_id", columns="controller")
    for metric in ("TBR", "LBGI"):
        smpc = pivot[metric]["stochastic_mpc"].to_numpy()
        dmpc = pivot[metric]["deterministic_mpc"].to_numpy()
        stat, pval = wilcoxon(smpc, dmpc, zero_method="pratt", alternative="less")
        diffs = smpc - dmpc
        delta = np.mean(diffs)
        ci_low, ci_high = bootstrap_ci(diffs)
        metrics[metric] = {
            "wilcoxon_stat": stat,
            "p_value": pval,
            "mean_difference": delta,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "rank_biserial": rank_biserial_from_wilcoxon(stat, len(diffs)),
        }
    stats_df = pd.DataFrame(metrics).T
    stats_df.index.name = "metric"
    stats_df.to_csv(STATS_CSV)
    return stats_df


def main():
    comparison_results = load_results()
    df = results_to_dataframe(comparison_results)
    patient_id = choose_representative_patient(df)
    plot_glucose_traces(comparison_results, patient_id)
    plot_cvga(df)
    stats_df = run_wilcoxon(df)
    print("Wilcoxon signed-rank (SMPC < DMPC):")
    print(stats_df)
    print(f"Representative patient for glucose trace: {patient_id}")
    print(f"\nSaved figures:\n - {FIG_GLUCOSE}\n - {FIG_CVGA}")
    print(f"Stats CSV: {STATS_CSV}")


if __name__ == "__main__":
    main()
