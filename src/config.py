# src/config.py

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class Config:
    """
    Centralized configuration for paths and constants.
    Update BASE_DIR to match your local project layout if needed.
    """
    BASE_DIR: Path = Path("data_challenge_in")

    DATA_SUBDIR: str = "data"
    DIM_SUBDIR: str = "dimension"
    OUT_SUBDIR: str = "outputs"
    FIG_SUBDIR: str = "outputs/figures"

    # Pre/Post split rule
    CUTOFF_YEAR: int = 2020
    CUTOFF_QUARTER: int = 1

    # File patterns
    ARFF_GLOB: str = "*.arff"

    # Thresholds
    MISSING_COL_DROP_THRESHOLD: float = 0.70

    # Column names
    COUNTRY_COL: str = "country"
    SECTOR_COL: str = "sector"

    def derived_paths(self) -> dict:
        return {
            "DATA_DIR": self.BASE_DIR / self.DATA_SUBDIR,
            "DIM_DIR": self.BASE_DIR / self.DIM_SUBDIR,
            "OUT_DIR": self.BASE_DIR / self.OUT_SUBDIR,
            "FIG_DIR": self.BASE_DIR / self.FIG_SUBDIR,
        }

def init_environment(cfg: Config):
    """Creates folders and sets display settings."""
    paths = cfg.derived_paths()

    # Create folders if they don't exist
    paths["OUT_DIR"].mkdir(parents=True, exist_ok=True)
    paths["FIG_DIR"].mkdir(parents=True, exist_ok=True)

    # Pandas display settings
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    # Logging
    print("✅ Environment Initialized")
    for name, path in paths.items():
        print(f" - {name}: {path}")
    print(f"Pre/Post cutoff at {cfg.CUTOFF_YEAR} Q{cfg.CUTOFF_QUARTER}")

    return paths
