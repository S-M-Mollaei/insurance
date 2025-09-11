import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

class CorrelationAnalyzer:
    def __init__(self, feature_prefix: str = "X", target_col: str = "S", verbose: bool = True, save_path: Optional[str] = None):
        self.feature_prefix = feature_prefix
        self.target_col = target_col
        self.verbose = verbose
        self.save_path = save_path

    def compute(self, df: pd.DataFrame, top_n: int = 10) -> pd.Series:
        """Compute absolute correlation of features with target and return top N."""
        feature_cols = [col for col in df.columns if col.startswith(self.feature_prefix)]
        correlations = df[feature_cols + [self.target_col]].corr()
        top_corr = correlations[self.target_col].drop(self.target_col).sort_values(key=abs, ascending=False).head(top_n)

        if self.verbose:
            print(f"🔝 Top {top_n} features most correlated with target ({self.target_col}):\n")
            for feature, corr in top_corr.items():
                print(f"{feature:<5} → correlation: {corr:.4f}")

        return top_corr

    def plot(self, top_corr: pd.Series, figsize: tuple = (10, 6)) -> None:
        """Plot a barplot of top correlations with target."""
        plt.figure(figsize=figsize)
        sns.barplot(x=top_corr.values, y=top_corr.index)
        plt.title(f"Top {len(top_corr)} Correlations with Target ({self.target_col})")
        plt.xlabel("Correlation")
        plt.ylabel("Feature")
        plt.tight_layout()
        if self.save_path:
            plt.savefig(f"{self.save_path}/top_correlations.png")
        print("✅ Correlation plot generated and saved.")
        plt.show()

    def clean_target(self, df: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """Drop rows with missing target values and optionally save to CSV."""
        before = df[self.target_col].isna().sum()
        if self.verbose:
            print(f"Before cleaning: {before} missing in {self.target_col}")

        df = df.dropna(subset=[self.target_col]).copy()

        after = df[self.target_col].isna().sum()
        if self.verbose:
            print(f"After cleaning: {after} missing in {self.target_col}")

        if save_path:
            df.to_csv(save_path, index=False)
            if self.verbose:
                print(f"✅ Cleaned data saved to {save_path}")

        return df
