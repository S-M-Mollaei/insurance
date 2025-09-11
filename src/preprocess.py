from pathlib import Path
import pandas as pd
from typing import List, Optional

class Preprocessor:
    def __init__(self, dim_dir: Path, verbose: bool = True):
        self.dim_dir = dim_dir
        self.verbose = verbose
        self.sector_map = {}

    def remove_violations(self, df: pd.DataFrame, violating_countries: Optional[List[str]] = None) -> pd.DataFrame:
        if violating_countries is None:
            violating_countries = ["Italy"]

        before = df.shape[0]
        df = df[~df["Country"].isin(violating_countries)].copy()
        after = df.shape[0]
        if self.verbose:
            print(f"🧹 Removed {before - after} rows from: {violating_countries}")
        return df

    def map_sector_names(self, df: pd.DataFrame) -> pd.DataFrame:
        sector_path = self.dim_dir / "sector_dimension.csv"
        sector_df = pd.read_csv(sector_path)

        self.sector_map = dict(zip(sector_df["code_sector"], sector_df["description_sector"]))
        df["sector_name"] = df["S"].map(self.sector_map)

        missing = df[df["sector_name"].isna()]["S"].unique()
        if self.verbose and len(missing) > 0:
            print("⚠️ Unmapped sector codes found:", missing)

        return df

    def summarize(self, df: pd.DataFrame, feature_prefix: str = "X", sample_cols: int = 10) -> None:
        feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]

        print("🔎 Data Overview:")
        print(" - Shape:", df.shape)
        print(" - First columns:", df.columns.tolist()[:sample_cols])
        print()
        print(df.info())
        print(df[feature_cols].describe())
        print()
        print("📊 Per-sector stats (X1 & X2):")
        print(df.groupby("sector_name")[feature_cols].describe()[["X1", "X2"]])

        return feature_cols
