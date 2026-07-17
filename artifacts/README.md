# Artifact policy

- `metrics/` contains small, reviewable CSV evaluation outputs and is suitable for
  version control.
- `plots/` contains reviewable evaluation figures.
- `models/` contains local serialized model outputs. Pickle files are intentionally
  ignored because they are environment-dependent and should be regenerated or stored
  in a model registry.
- `training_logs/` is local runtime output and is ignored.

For production use, package the estimator together with preprocessing, encoders,
the ordered feature schema, training-data version, and evaluation metadata.
