"""Generate leakage-safe historical features from the canonical raw dataset."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ufc_prediction.features.history import generate_engineered_dataset


if __name__ == "__main__":
    generate_engineered_dataset()
