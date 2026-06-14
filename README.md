# Padova SMPC Benchmark

This repository contains the complete benchmark used to evaluate stochastic MPC (SMPC) against deterministic MPC (DMPC) and a PID baseline for closed-loop insulin delivery in a preclinical in-silico setting. The study tests whether explicit uncertainty propagation through chance constraints improves glycemic outcomes when the surrogate model, estimator, prediction horizon, and pump limits are held identical across both controllers.

---

## Contents

```
├── run_experiments.py          # Main simulation runner (900 subject-days per controller)
├── digital_twin.py             # 9-state Padova-inspired stochastic surrogate
├── controllers_improved.py     # SMPC, DMPC, and PID controllers (OSQP backend)
├── kalman_filter.py            # Extended Kalman Filter for state estimation
├── padova_synthetic.py         # Synthetic cohort generation (adults, adolescents, children)
├── scenarios.py                # Scenario definitions and patient simulation logic
├── metrics.py                  # Glycemic metrics (TIR, TBR, TAR, LBGI, HBGI, CV)
├── config.py                   # All hyperparameters and constants
├── train_test_split.py         # Prospective train/test partition definition
│
├── analyze_all_results.py      # Aggregate results, statistical tests, Table generation
├── analyze_nocturnal_child.py  # Focused nocturnal child analysis (Figure 9)
├── build_manuscript_assets.py  # All publication figures (Figures 1–9)
├── run_targeted_ablation.py    # Ablation study (Table 9)
├── tune_hyperparameters.py     # Grid-search tuning on held-out training subjects
├── profile_runtime.py          # Solver timing benchmarks
│
├── legacy/
│   └── controllers_cvxpy_legacy.py  # Deprecated CVXPY-based prototype (not used for results)
│
├── results/
│   ├── data/                   # Processed result tables (CSV, JSON)
│   │   └── README.md           # Description of each file
│   └── figures/                # Publication figures (Figures 1–9) and high-res variants
│
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Setup

### Requirements

Python 3.10 or 3.11 recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `scipy`, `osqp`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm`.

> **Note on `cvxpy`:** The production controllers in `controllers_improved.py` use OSQP directly. `cvxpy` is only needed if you want to run the legacy prototype in `legacy/controllers_cvxpy_legacy.py`.

### Reproducing results from scratch

All simulations are seeded with `PADOVA_SEED = 42` in `config.py`. Outputs are fully deterministic.

**Step 1 — Run the main experiments**

```bash
python run_experiments.py
```

This runs 100 subjects × 3 cohorts × 3 scenarios × 3 controllers = 2700 simulations (approximately 900 subject-days per controller). Progress is shown with tqdm. Results are saved as pickle files in `results/data/`.

Expected runtime: 30–90 minutes depending on hardware (see `results/data/runtime_profile_summary.csv` for timing benchmarks).

**Step 2 — Analyze results and generate tables**

```bash
python analyze_all_results.py
```

Reads the pickle files, computes glycemic metrics, runs Wilcoxon signed-rank tests with Benjamini-Hochberg correction across 63 paired tests, and writes all CSV and JSON result tables to `results/data/`.

**Step 3 — Run the ablation study**

```bash
python run_targeted_ablation.py
```

Runs four controller variants (full SMPC, no-variance, no-chance-tightening, short-horizon) on the child cohort under variable and nocturnal scenarios.

**Step 4 — Generate figures**

```bash
python build_manuscript_assets.py
```

Generates all publication figures (Figures 1–8) in `results/figures/`. Figure 9 is generated separately:

```bash
python analyze_nocturnal_child.py
```

---

## Surrogate model

The simulation environment is a 9-state ordinary differential equation system inspired by the UVA/Padova Type 1 Diabetes Simulator. It models subcutaneous insulin absorption (two compartments), plasma insulin, insulin action, gut absorption (two compartments), plasma glucose, interstitial glucose (CGM), and an Ornstein-Uhlenbeck insulin sensitivity multiplier. Stochastic process noise enters through all state equations; parameter values are calibrated to match published mean ± SD cohort summaries for 100 virtual subjects per age group.

The same surrogate acts as both the simulated plant and the controller's internal model. This gives the EKF privileged structural knowledge that would not hold against a real patient. The primary claims of the paper are therefore methodological, not clinical.

---

## Controller architecture

**Stochastic MPC (SMPC):** At each 5-minute control interval, the EKF provides a state estimate and posterior covariance. The controller propagates predictive glucose mean and covariance forward over a 12-step (60-minute) horizon. Gaussian chance constraints at the 1% hypo and 10% hyper risk levels are converted to deterministic glucose guardrails using precomputed z-score factors (κ = 2.326 and κ = 1.282). The resulting quadratic program is solved with OSQP.

**Deterministic MPC (DMPC):** Identical controller, estimator, horizon, pump limits, and cost function weights. The only difference is that chance-constraint tightening is disabled and predictive variance is not penalized — the controller uses fixed glucose bounds.

**PID:** Baseline proportional-integral-derivative controller with cohort-specific gains from `config.py`.

---

## Cohorts and scenarios

Three age cohorts (adult, adolescent, child), each with 100 synthetic subjects generated using `padova_synthetic.py`. Three scenarios:

| Scenario | Description |
|----------|-------------|
| Standard | Three announced meals at fixed times. Meal sizes in g/kg body weight: breakfast 0.67, lunch 0.93, dinner 0.80. |
| Variable | Random meal timing (±1 h), random meal sizes (0.40–1.20 g/kg), 30% chance of an unannounced afternoon snack. |
| Nocturnal | Same as standard but dinner and a late-night snack (0.24 g/kg at 22:00) are unannounced. |

Meal sizes are specified as g/kg relative to body weight. See `config.py` → `MEAL_SCENARIOS_GPKG` for exact values.

---

## Train/test partition

Hyperparameters were tuned exclusively on the first 30 adult subjects (indices 0–29, seed 42) under the standard scenario. All 100 adult subjects, all 100 adolescent subjects, and all 100 child subjects were then evaluated on the held-out test set. See `train_test_split.py` for the exact split definition and `tune_hyperparameters.py` for the grid-search procedure.

---

## Result files

Processed tables are in `results/data/`. See `results/data/README.md` for a description of each file.

Raw per-subject simulation outputs (pickle files, ~15 MB total) are not included in this repository. They can be regenerated with `run_experiments.py` (Step 1 above) or requested from the corresponding author.

---

## Figures

Publication figures are in `results/figures/`. The manuscript LaTeX uses `figure1_framework_schematic_new.png` and `figure2_nine_state_model_new.png` (higher-resolution variants of Figures 1 and 2a); all other figures use the standard filenames.

| File | Manuscript figure |
|------|------------------|
| `figure1_framework_schematic_new.png` | Figure 1 — Full benchmark pipeline |
| `figure2_nine_state_model_new.png` | Figure 2a — Nine-state model diagram |
| `figure2_surrogate_sanity_check.png` | Figure 2b — Surrogate sanity checks |
| `figure3_chance_constraints.png` | Figure 3 — Chance-constraint geometry |
| `figure4_scenario_timeline.png` | Figure 4 — Scenario meal timelines |
| `figure5_nocturnal_child_trace_insulin.png` | Figure 5 — Representative nocturnal child trace |
| `figure6_outcome_distributions.png` | Figure 6 — Outcome distributions across cohorts |
| `figure7_forest_plot.png` | Figure 7 — Forest plot of effect sizes |
| `figure8_pediatric_cvga_panel.png` | Figure 8 — Pediatric CVGA panel |
| `figure9_nocturnal_child_cvga.png` | Figure 9 — Nocturnal child CVGA |

---

## Citation

If you use this code or data, please cite:

> [Citation to be added upon publication]

---

## License

GNU General Public License. See `LICENSE`.

---

## Data availability

The processed result tables underlying all primary findings are included in `results/data/`. Raw per-subject simulation outputs are available from the corresponding author upon reasonable request. The synthetic patient cohort is fully reproducible from `padova_synthetic.py` with `PADOVA_SEED = 42`.
