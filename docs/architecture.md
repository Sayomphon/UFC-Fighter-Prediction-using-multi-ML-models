# Architecture

The project is organized around a temporal, leakage-aware modeling flow:

```text
data/raw/ufc-master.csv
        |
        +--> canonical 40-feature list ----------------------+
        |                                                    |
        +--> optional historical feature builder             |
        |    (Elo, form, finish rate, layoff, experience)     |
        |                                                    v
        +--> train-period imputation --> encoding --> scaling --> model training
                                                               |
                                                               v
                    winner / method / two-stage round probabilities
                                                               |
                                                               v
                    holdout + rolling-origin evaluation
                                                               |
                          artifacts/metrics + artifacts/plots
```

## Boundaries

- `src/ufc_prediction` is the reusable implementation boundary.
- `scripts` contains runnable orchestration entrypoints.
- `notebooks` contains experiment narratives and historical Colab exports.
- `data` contains canonical inputs and derived tabular data.
- `artifacts` contains generated outputs; large or environment-dependent files should
  be ignored or managed by DVC/MLflow rather than treated as source code.

## Current limitations

- The end-to-end training pipeline remains notebook-centric.
- Serialized models do not yet bundle preprocessing, encoders, or schema metadata.
- There is no inference API, drift monitoring, model registry, or CI workflow yet.
- Rolling-origin sportsbook log loss must be compared on the same odds-available rows
  before making a production superiority claim.
