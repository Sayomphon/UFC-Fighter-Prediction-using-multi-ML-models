# UFC Fighter Prediction

Leakage-aware machine-learning experiments for predicting UFC fight winner, method,
and round probabilities. The repository separates reusable code from notebooks,
datasets, and generated artifacts so there is one clear path through the project.

## Current status

- **Winner:** CatBoost is the strongest holdout model. On the 229-fight real-odds
  subset it reached 0.7118 accuracy and 0.5748 log loss, versus 0.7074 and 0.5880
  for vig-removed sportsbook probabilities.
- **Stability:** rolling-origin validation (2020-2024) does not yet show a consistent
  advantage over the favorite-pick baseline.
- **Method:** the three-class KO/TKO, SUB, DEC model improves macro-F1 over a
  class-prior baseline.
- **Round:** the task is split into finish-vs-decision and R1/R2/R3+ conditional on
  a finish.
- **Production readiness:** this remains an evaluated research prototype, not a
  deployed betting or decision system.

Detailed metrics are stored in [`artifacts/metrics`](artifacts/metrics), and the
calibration plot is in [`artifacts/plots`](artifacts/plots).

## Repository map

```text
data/                  Canonical raw, processed, and feature-list datasets
src/ufc_prediction/    Reusable Python package: paths, features, and metrics
scripts/               Runnable project entrypoints
notebooks/             Experiments ordered from baseline to evaluation
artifacts/              Models, metrics, plots, and local training logs
docs/                   Architecture, feature rationale, and project documentation
tests/                  Fast tests for paths, metrics, and leakage-safe features
```

The exported Python files named `legacy_export.py` are retained only as experiment
history. They are Colab exports and are not the canonical reusable implementation.

## Quick start

Use Python 3.9 or newer from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Generate the leakage-safe historical feature dataset:

```bash
python scripts/build_features.py
```

Run the full evaluation harness (this retrains CatBoost models and can take time):

```bash
python scripts/evaluate.py
```

Run the fast verification suite:

```bash
pytest
```

## Data and evaluation policy

- Canonical raw data: [`data/raw/ufc-master.csv`](data/raw/ufc-master.csv)
- Selected model inputs: [`data/features/selected_features_40.csv`](data/features/selected_features_40.csv)
- Training cutoff: `2024-03-31`
- Holdout: fights after the cutoff
- Rolling-origin folds: 2020 through 2024
- Post-fight fields such as winner, finish details, finish round, and total fight time
  are excluded from model inputs.
- Imputation statistics are recomputed from the training period for each fold.

See [`docs/architecture.md`](docs/architecture.md) for the system flow and
[`docs/feature_engineering.md`](docs/feature_engineering.md) for the history-feature
ablation and its limitations.

## Notebook order

1. [`02_baseline`](notebooks/02_baseline) — Random Forest baseline
2. [`03_feature_selection`](notebooks/03_feature_selection) — feature importance and RFE
3. [`04_model_comparison`](notebooks/04_model_comparison) — ML/DL comparison
4. [`05_hyperparameter_tuning`](notebooks/05_hyperparameter_tuning) — tuning experiments
5. [`06_pipeline`](notebooks/06_pipeline) — end-to-end research pipeline
6. [`07_evaluation`](notebooks/07_evaluation) — standard evaluation framework

Launch Jupyter from the repository root so legacy notebook-relative paths resolve
against the canonical `data/` and `artifacts/` directories.

## Data sources

The original project used the
[UFC Web Scraping repository](https://github.com/remypereira99/UFC-Web-Scraping)
and the [Ultimate UFC Dataset on Kaggle](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)
as starting points. Review source licensing and refresh policy before redistribution
or production use.
