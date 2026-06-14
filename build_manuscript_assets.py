from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

import config


FIG_DIR = Path(config.FIGURES_DIR)
DATA_DIR = Path(config.DATA_DIR)


def load_all_results() -> Dict[str, Dict[str, List[Dict]]]:
    all_results: Dict[str, Dict[str, List[Dict]]] = {}
    for scenario in config.MEAL_SCENARIOS:
        all_results[scenario] = {}
        for cohort in config.COHORTS:
            path = DATA_DIR / f"results_{scenario}_{cohort}.pkl"
            with path.open("rb") as handle:
                all_results[scenario][cohort] = pickle.load(handle)
    return all_results


def results_to_metrics_df(all_results: Dict[str, Dict[str, List[Dict]]]) -> pd.DataFrame:
    rows = []
    for scenario, cohorts in all_results.items():
        for cohort, result_dict in cohorts.items():
            for controller, runs in result_dict.items():
                for run in runs:
                    row = {
                        "scenario": scenario,
                        "cohort": cohort,
                        "controller": controller,
                        "patient_id": run["patient_id"],
                    }
                    row.update(run["metrics"])
                    rows.append(row)
    return pd.DataFrame(rows)


def choose_representative_patient(df: pd.DataFrame, scenario: str, cohort: str) -> int:
    sub = df[(df["scenario"] == scenario) & (df["cohort"] == cohort)]
    patient_ids = np.sort(sub["patient_id"].unique())
    rng = np.random.default_rng(42)
    return int(rng.choice(patient_ids))


def _add_box(ax, xy, text, width=1.8, height=0.75, fc="#F7F4EA", ec="#1F2933"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)
    return box


def _arrow(ax, start, end, text=None, y_offset=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#1F2933"),
    )
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + y_offset, text, fontsize=9, ha="center")


def plot_framework_schematic() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    _add_box(ax, (0.35, 3.85), "Synthetic Subject\nPlant", width=1.55, fc="#FDEBD0")
    _add_box(ax, (2.3, 3.85), "CGM Sensing\nNoise + Delay", width=1.55, fc="#D6EAF8")
    _add_box(ax, (4.25, 3.85), "EKF State\nEstimation", width=1.55, fc="#D5F5E3")
    _add_box(ax, (6.2, 3.85), "Stochastic Twin\nPrediction", width=1.55, fc="#FCF3CF")
    _add_box(ax, (8.15, 3.85), "Gaussian Uncertainty\nPropagation", width=1.8, fc="#E8DAEF")
    _add_box(ax, (10.45, 3.85), "Chance-Constrained\nSMPC", width=1.5, fc="#FADBD8")
    _add_box(ax, (10.45, 1.35), "Insulin Pump\nActuation", width=1.5, fc="#FDEDEC")
    _add_box(ax, (2.0, 1.35), "Meals\nAnnounced + Hidden", width=1.85, fc="#FEF5E7")

    _arrow(ax, (1.9, 4.22), (2.3, 4.22))
    ax.text(2.1, 4.48, "interstitial glucose", fontsize=8.3, ha="center")
    _arrow(ax, (3.85, 4.22), (4.25, 4.22))
    ax.text(4.05, 4.48, "delayed CGM", fontsize=8.3, ha="center")
    _arrow(ax, (5.8, 4.22), (6.2, 4.22))
    ax.text(6.0, 4.48, "state mean / covariance", fontsize=8.3, ha="center")
    _arrow(ax, (7.75, 4.22), (8.15, 4.22))
    ax.text(7.95, 4.48, "forecast variance", fontsize=8.3, ha="center")
    _arrow(ax, (9.95, 4.22), (10.45, 4.22))
    ax.text(10.2, 4.48, "tightened guardrails", fontsize=8.3, ha="center")

    _arrow(ax, (11.2, 3.85), (11.2, 2.1))
    ax.text(11.45, 3.0, "insulin command", rotation=90, fontsize=8.3, va="center")
    _arrow(ax, (10.45, 1.72), (1.9, 3.98))
    ax.text(6.2, 2.9, "pump actuation closes the loop", fontsize=8.3, ha="center")
    _arrow(ax, (3.85, 1.72), (6.2, 3.85))
    ax.text(5.1, 2.5, "announced meals inform prediction", fontsize=8.0, ha="center")
    _arrow(ax, (2.7, 1.72), (1.1, 3.85))
    ax.text(1.8, 2.5, "hidden events perturb the plant", fontsize=8.0, ha="center")

    ax.text(6.25, 5.8, "Closed-Loop Architecture: Stochastic Twin, Estimation, and Risk-Aware Control", ha="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_framework_schematic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nine_state_model() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    coords = {
        "q_sto1": (0.5, 2.4),
        "q_sto2": (2.4, 2.4),
        "q_gut": (4.3, 2.4),
        "G_p": (6.2, 3.2),
        "G_i": (8.1, 3.2),
        "I_sc1": (6.2, 1.3),
        "I_sc2": (8.1, 1.3),
        "I_p": (10.0, 1.3),
        "X": (10.0, 3.2),
    }
    labels = {
        "q_sto1": "q_sto1\nStomach solid\n(mg)",
        "q_sto2": "q_sto2\nStomach liquid\n(mg)",
        "q_gut": "q_gut\nIntestine\n(mg)",
        "G_p": "G_p\nPlasma glucose\n(mg/dL)",
        "G_i": "G_i\nInterstitial glucose\n(mg/dL)",
        "I_sc1": "I_sc1\nSC insulin 1\n(mU)",
        "I_sc2": "I_sc2\nSC insulin 2\n(mU)",
        "I_p": "I_p\nPlasma insulin\n(mU/L)",
        "X": "X\nInsulin action\n(mU/L)",
    }
    palette = {
        "meal": "#FDEBD0",
        "glucose": "#D6EAF8",
        "insulin": "#FADBD8",
    }

    for key, (x_pos, y_pos) in coords.items():
        if key.startswith("q_"):
            fc = palette["meal"]
        elif key.startswith("G"):
            fc = palette["glucose"]
        else:
            fc = palette["insulin"]
        _add_box(ax, (x_pos, y_pos), labels[key], width=1.45, height=1.0, fc=fc)

    _arrow(ax, (1.95, 2.9), (2.4, 2.9))
    _arrow(ax, (3.85, 2.9), (4.3, 2.9))
    _arrow(ax, (5.75, 2.9), (6.2, 3.45), "Ra", y_offset=0.1)
    _arrow(ax, (7.65, 3.65), (8.1, 3.65))
    _arrow(ax, (7.65, 1.8), (8.1, 1.8))
    _arrow(ax, (9.55, 1.8), (10.0, 1.8))
    _arrow(ax, (10.7, 2.3), (10.7, 3.2), "insulin effect", y_offset=0.1)
    _arrow(ax, (10.0, 3.7), (7.65, 3.7))
    ax.text(8.95, 3.95, "glucose utilization", fontsize=8.3, ha="center")
    ax.text(8.25, 3.42, "CGM output", fontsize=8.3, ha="center")
    ax.text(0.6, 4.85, "Meal absorption", fontsize=11, fontweight="bold")
    ax.text(6.3, 4.85, "Glucose subsystem", fontsize=11, fontweight="bold")
    ax.text(6.3, 0.45, "Insulin subsystem", fontsize=11, fontweight="bold")
    ax.text(6.5, 5.25, "Nine-State Physiologically Inspired Surrogate Model", ha="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_nine_state_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_chance_constraint_concept() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    t = np.arange(config.MPC_HORIZON) * config.SAMPLING_TIME / 60.0
    mean = 118 - 10 * np.sin(np.linspace(0, 2.5, len(t))) - np.linspace(0, 18, len(t))
    sigma = 7 + 10 * np.exp(-((t - 0.7) ** 2) / 0.12)
    lower_guard = config.G_MIN + config.KAPPA_HYPO * sigma
    upper_guard = config.G_MAX - config.KAPPA_HYPER * sigma
    deterministic_guard = np.full_like(t, config.G_MIN)

    ax.fill_between(t, mean - sigma, mean + sigma, color="#D6EAF8", alpha=0.65, label="SMPC predictive $\\mu_G \\pm \\sigma_G$")
    ax.plot(t, mean, color="#2E86AB", linewidth=2.2, label="SMPC predictive mean")
    ax.plot(t, lower_guard, color="#A23B72", linestyle="--", linewidth=2.0, label="Tightened lower guardrail")
    ax.plot(t, upper_guard, color="#AF601A", linestyle="--", linewidth=2.0, label="Tightened upper guardrail")
    ax.plot(t, deterministic_guard, color="#566573", linestyle=":", linewidth=2.0, label="Deterministic lower threshold")
    ax.axhline(config.G_MIN, color="#566573", linestyle=":", linewidth=1.3)
    ax.axhline(config.G_MAX, color="#7B7D7D", linestyle=":", linewidth=1.3)
    ax.text(t[-1] + 0.05, config.G_MIN, "70 mg/dL", va="center", fontsize=9)
    ax.text(t[-1] + 0.05, config.G_MAX, "250 mg/dL", va="center", fontsize=9)
    ax.set_xlabel("Prediction horizon (hours)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_title("Chance Constraints Convert Variance into Dynamic Glucose Guardrails", fontweight="bold")
    ax.set_ylim(40, 280)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_chance_constraints.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_timeline() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 28)
    ax.set_ylim(-0.8, 2.8)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Standard", "Variable", "Nocturnal"])
    ax.set_xlabel("Simulation clock (hours)")
    ax.set_title("Scenario Timelines, Warmup, and Disturbance Disclosure", fontweight="bold")

    ax.axvspan(0, config.WARMUP_HOURS, color="#EAECEE", alpha=0.8)
    ax.axvspan(config.WARMUP_HOURS, 28, color="#FCFDFD", alpha=0.3)
    ax.text(2, 2.55, "Warmup\n4 h", ha="center", fontsize=10)
    ax.text(16, 2.55, "Evaluation window\n24 h", ha="center", fontsize=10)

    # Standard scenario — g/kg labels (absolute gram shown in parentheses for 75 kg ref)
    base_meals = [(7, 0.67, True), (12, 0.93, True), (18, 0.80, True)]
    for y in [2, 1, 0]:
        ax.hlines(y, 0, 28, color="#ABB2B9", linewidth=1.2)

    for hour, gpkg, announced in base_meals:
        style = "-" if announced else "--"
        ax.vlines(hour, 1.82, 2.18, color="#2E86AB", linewidth=2.5, linestyles=style)
        ax.text(hour, 2.28, f"{gpkg} g/kg", ha="center", fontsize=9)

    ax.add_patch(Rectangle((6, 0.82), 2, 0.36, facecolor="#D6EAF8", edgecolor="#2E86AB", alpha=0.8))
    ax.add_patch(Rectangle((11, 0.82), 2, 0.36, facecolor="#D6EAF8", edgecolor="#2E86AB", alpha=0.8))
    ax.add_patch(Rectangle((17, 0.82), 2, 0.36, facecolor="#D6EAF8", edgecolor="#2E86AB", alpha=0.8))
    ax.text(7, 1.28, "breakfast\n0.40–1.20 g/kg", ha="center", fontsize=8)
    ax.text(12, 1.28, "lunch\n0.40–1.20 g/kg", ha="center", fontsize=8)
    ax.text(18, 1.28, "dinner\n0.40–1.20 g/kg", ha="center", fontsize=8)
    ax.vlines(15, 0.82, 1.18, color="#A23B72", linewidth=2.5, linestyles="--")
    ax.text(15, 1.28, "hidden snack\n0.13–0.27 g/kg, p=0.3", ha="center", fontsize=8)

    nocturnal_meals = [(7, 0.67, True), (12, 0.93, True), (18, 0.80, False), (22, 0.24, False)]
    for hour, gpkg, announced in nocturnal_meals:
        color = "#2E86AB" if announced else "#A23B72"
        style = "-" if announced else "--"
        ax.vlines(hour, -0.18, 0.18, color=color, linewidth=2.5, linestyles=style)
        tag = "announced" if announced else "hidden"
        ax.text(hour, 0.28, f"{gpkg} g/kg\n{tag}", ha="center", fontsize=8)

    ax.text(26.2, 2.1, f"CGM delay: {config.CGM_DELAY_STEPS * config.SAMPLING_TIME} min", fontsize=9)
    ax.text(26.2, 1.8, f"Horizon: {config.MPC_HORIZON * config.SAMPLING_TIME} min", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_scenario_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_representative_trace(all_results: Dict[str, Dict[str, List[Dict]]], patient_id: int) -> None:
    comparison = all_results["nocturnal"]["child"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    colors = {"stochastic_mpc": "#2E86AB", "deterministic_mpc": "#A23B72"}
    labels = {"stochastic_mpc": "SMPC", "deterministic_mpc": "DMPC"}

    meal_schedule = None
    meal_announcements = None
    for controller in ("stochastic_mpc", "deterministic_mpc"):
        run = next(r for r in comparison[controller] if r["patient_id"] == patient_id)
        glucose = np.asarray(run["glucose"])
        insulin = np.asarray(run["insulin"])
        time_hours = np.arange(len(glucose)) * config.SAMPLING_TIME / 60.0
        axes[0].plot(time_hours, glucose, color=colors[controller], linewidth=1.9, label=labels[controller])
        axes[1].step(time_hours, insulin * 60.0, where="post", color=colors[controller], linewidth=1.7, label=labels[controller])
        meal_schedule = run["meal_schedule"]
        meal_announcements = run["meal_announcements"]

    axes[0].axhspan(config.TIR_LOWER, config.TIR_UPPER, color="#E8F6EF", alpha=0.55, label="Target 70-180 mg/dL")
    axes[0].axhline(config.G_SEVERE_HYPO, color="#C0392B", linestyle="--", linewidth=1.2, label="54 mg/dL")

    for step, cho in meal_schedule.items():
        hour = step * config.SAMPLING_TIME / 60.0
        announced = meal_announcements.get(step, True)
        color = "#1F618D" if announced else "#922B21"
        style = "-" if announced else "--"
        axes[0].axvline(hour, color=color, linestyle=style, linewidth=1.3, alpha=0.8)
        axes[1].axvline(hour, color=color, linestyle=style, linewidth=1.0, alpha=0.7)
        axes[0].text(hour, 247, f"{cho:.0f} g", rotation=90, ha="center", va="top", fontsize=8, color=color)

    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].set_ylim(40, 260)
    axes[0].set_title(f"Fixed Random Nocturnal Child Pairing (patient {patient_id})", fontweight="bold")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].set_ylabel("Insulin\n(U/h)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_nocturnal_child_trace_insulin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_distributions(metrics_df: pd.DataFrame) -> None:
    df = metrics_df[metrics_df["controller"].isin(["stochastic_mpc", "deterministic_mpc"])].copy()
    df["controller"] = df["controller"].map({"stochastic_mpc": "SMPC", "deterministic_mpc": "DMPC"})
    metric_map = [("TIR", "TIR (%)"), ("TBR", "TBR (%)"), ("LBGI", "LBGI")]
    cohort_order = ["adult", "adolescent", "child"]
    scenario_order = ["standard", "variable", "nocturnal"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    palette = {"SMPC": "#2E86AB", "DMPC": "#A23B72"}
    for row, (metric, ylabel) in enumerate(metric_map):
        for col, cohort in enumerate(cohort_order):
            ax = axes[row, col]
            sub = df[df["cohort"] == cohort]
            sns.boxplot(
                data=sub,
                x="scenario",
                y=metric,
                hue="controller",
                order=scenario_order,
                palette=palette,
                width=0.7,
                fliersize=3.5,
                linewidth=1.0,
                ax=ax,
            )
            if row == 0:
                ax.set_title(cohort.title(), fontweight="bold", fontsize=13)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=10)
            ax.grid(axis="y", alpha=0.2)
            if row == 0 and col == 2:
                ax.legend(title=None, loc="upper right", fontsize=10)
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
    fig.suptitle("Cohort-Level Distributions of Core Endpoints", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "figure6_outcome_distributions.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_forest() -> None:
    df = pd.read_csv(DATA_DIR / "statistical_comparison_summary.csv")
    metrics = [
        ("TIR", "TIR (%)"),
        ("TBR", "TBR (%)"),
        ("Time_below_54", "Time <54 (%)"),
        ("LBGI", "LBGI"),
        ("CV", "CV (%)"),
        ("Mean_glucose", "Mean glucose (mg/dL)"),
    ]
    scenario_order = {"standard": 0, "variable": 1, "nocturnal": 2}
    cohort_order = {"adult": 0, "adolescent": 1, "child": 2}

    fig, axes = plt.subplots(2, 3, figsize=(14, 11))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metrics):
        sub = df[df["metric"] == metric].copy()
        sub["sort_key"] = sub["scenario"].map(scenario_order) * 3 + sub["cohort"].map(cohort_order)
        sub = sub.sort_values("sort_key", ascending=False)
        y = np.arange(len(sub))
        ax.axvline(0, color="#566573", linestyle=":", linewidth=1.2)
        ax.errorbar(
            sub["mean_difference"],
            y,
            xerr=[sub["mean_difference"] - sub["ci95_low"], sub["ci95_high"] - sub["mean_difference"]],
            fmt="o",
            color="#2E86AB",
            ecolor="#2E86AB",
            elinewidth=1.5,
            capsize=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([f"{s.title()} | {c.title()}" for s, c in zip(sub["scenario"], sub["cohort"])], fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("SMPC - DMPC")
    fig.suptitle("Paired Effect Estimates with 95% Confidence Intervals", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "figure7_forest_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pediatric_cvga_panel(metrics_df: pd.DataFrame) -> None:
    child = metrics_df[(metrics_df["cohort"] == "child") & (metrics_df["controller"].isin(["stochastic_mpc", "deterministic_mpc"]))]
    scenarios = ["standard", "variable", "nocturnal"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharex=True, sharey=True)
    palette = {"stochastic_mpc": "#2E86AB", "deterministic_mpc": "#A23B72"}
    labels = {"stochastic_mpc": "SMPC", "deterministic_mpc": "DMPC"}

    for ax, scenario in zip(axes, scenarios):
        sub = child[child["scenario"] == scenario]
        for controller in ["stochastic_mpc", "deterministic_mpc"]:
            ctrl = sub[sub["controller"] == controller]
            ax.scatter(
                ctrl["Mean_glucose"],
                ctrl["Std_glucose"],
                c=palette[controller],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.65,
                s=35,
                label=labels[controller],
            )
        ax.add_patch(Rectangle((90, 0), 50, 45, facecolor="#ABEBC6", edgecolor="#52BE80", alpha=0.25))
        ax.axvline(config.TIR_LOWER, color="#7B7D7D", linestyle="--", linewidth=1)
        ax.axvline(config.TIR_UPPER, color="#7B7D7D", linestyle="--", linewidth=1)
        ax.axhline(50, color="#7B7D7D", linestyle=":", linewidth=1)
        ax.set_title(scenario.title(), fontweight="bold", fontsize=13)
        ax.set_xlabel("Mean glucose (mg/dL)", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Std. dev. glucose (mg/dL)", fontsize=13)
    axes[0].set_xlim(70, 200)
    axes[0].set_ylim(0, 130)
    axes[-1].legend(loc="upper right", fontsize=10)
    fig.suptitle("Pediatric Mean-Variability Comparison Across All Scenarios", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "figure8_pediatric_cvga_panel.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_all_results()
    metrics_df = results_to_metrics_df(all_results)
    patient_id = choose_representative_patient(metrics_df, "nocturnal", "child")

    plot_framework_schematic()
    plot_nine_state_model()
    plot_chance_constraint_concept()
    plot_scenario_timeline()
    plot_representative_trace(all_results, patient_id)
    plot_outcome_distributions(metrics_df)
    plot_forest()
    plot_pediatric_cvga_panel(metrics_df)

    print(f"Representative patient for Figure 5: {patient_id}")
    print("Saved manuscript figures to:")
    for name in [
        "figure1_framework_schematic.png",
        "figure2_nine_state_model.png",
        "figure3_chance_constraints.png",
        "figure4_scenario_timeline.png",
        "figure5_nocturnal_child_trace_insulin.png",
        "figure6_outcome_distributions.png",
        "figure7_forest_plot.png",
        "figure8_pediatric_cvga_panel.png",
    ]:
        print(f" - {FIG_DIR / name}")


if __name__ == "__main__":
    main()