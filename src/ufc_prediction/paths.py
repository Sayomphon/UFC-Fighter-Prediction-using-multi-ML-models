"""Canonical project paths shared by scripts and reusable modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
TRAINING_LOGS_DIR = ARTIFACTS_DIR / "training_logs"

RAW_FIGHTS_PATH = RAW_DATA_DIR / "ufc-master.csv"
ENGINEERED_FIGHTS_PATH = PROCESSED_DATA_DIR / "ufc-master-fe.csv"
SELECTED_FEATURES_PATH = FEATURES_DIR / "selected_features_40.csv"


def ensure_artifact_directories() -> None:
    """Create local output directories used by training and evaluation scripts."""

    for directory in (MODEL_DIR, METRICS_DIR, PLOTS_DIR, TRAINING_LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
