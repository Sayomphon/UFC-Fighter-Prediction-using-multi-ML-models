# Data layout

- `raw/ufc-master.csv` is the canonical source table used by the experiments.
- `processed/ufc-master-fe.csv` adds leakage-safe historical features.
- `processed/filtered_data_40_features.csv` is a legacy feature-selection output.
- `features/selected_features_40.csv` is the canonical ordered feature list used by
  winner, method, and round evaluation.

The raw table currently covers fights from 2010-03-21 through 2024-11-09. Data
refreshes should preserve the source schema and be validated before replacing the
canonical file. Do not derive pre-fight inputs from post-fight outcome columns.
