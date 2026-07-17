# Notebook guide

Notebooks record the experiment history. Reusable logic belongs in
`src/ufc_prediction`; new notebooks should import that package instead of copying
preprocessing and metric functions.

Run Jupyter from the repository root and follow this order:

1. `02_baseline/`
2. `03_feature_selection/`
3. `04_model_comparison/`
4. `05_hyperparameter_tuning/`
5. `06_pipeline/`
6. `07_evaluation/`

Files named `legacy_export.py` or `legacy_*_export.py` are Colab-generated snapshots.
They are retained for traceability and are not supported as standalone Python scripts.
