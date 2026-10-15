"""Shared project paths, resolved independently of the current directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_RAW_DATASET = RAW_DATA_DIR / "Raw_ISNA_Dataset.csv"
DEFAULT_CLEAN_DATASET = PROCESSED_DATA_DIR / "Clean_ISNA_Dataset.csv"


def ensure_project_directories() -> None:
    """Create directories used for data and generated reports."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
