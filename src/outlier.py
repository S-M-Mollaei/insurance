import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import List, Dict

class OutlierDetector:
    def __init__(self, feature_prefix: str = "X", threshold: float = 3.0, verbose: bool = True):
        self.feature_prefix = feature_prefix
        self.threshold = threshold
        self.verbose = verbose

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute z-scores and summarize outliers per feature."""
        feature_cols = [col for col in df.columns if col.startswith(self.feature_prefix)]
        z_scores = df[feature_cols].apply(zscore)

        outlier_report: Dict[str, Dict] = {}

        for col in feature_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()

            mask_outliers = z_scores[col].abs() > self.threshold
            outlier_values = df.loc[mask_outliers, col].tolist()

            outlier_report[col] = {
                "mean": col_mean,
                "std": col_std,
                "num_outliers": mask_outliers.sum(),
                "outlier_values": outlier_values[:10],  # preview first 10
            }

        outlier_summary = pd.DataFrame(outlier_report).T.sort_values("num_outliers", ascending=False)

        if self.verbose:
            print(f"🔎 Outlier summary (top 10 features with threshold |z| > {self.threshold}):")
            print(outlier_summary.head(10))

        return outlier_summary
