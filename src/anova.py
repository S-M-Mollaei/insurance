# src/anova.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from typing import List, Optional, Dict
from collections import Counter
from IPython.display import display


class ANOVAAnalyzer:
    def __init__(self, group_col: str = "period", verbose: bool = True, save_path: Optional[str] = None):
        self.group_col = group_col
        self.verbose = verbose
        self.save_path = save_path

    def run_global(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Run ANOVA for each feature between pre/post periods (globally)."""
        results = []

        for feature in feature_cols:
            pre_values = df[df[self.group_col] == "pre"][feature].dropna()
            post_values = df[df[self.group_col] == "post"][feature].dropna()

            if len(pre_values) < 2 or len(post_values) < 2:
                continue

            # ❗ Skip if both groups are constant
            if pre_values.nunique() <= 1 and post_values.nunique() <= 1:
                continue

            if len(pre_values) > 1 and len(post_values) > 1:
                f_stat, p_val = f_oneway(pre_values, post_values)
                results.append({
                    "feature": feature,
                    "F_statistic": f_stat,
                    "p_value": p_val
                })

        anova_df = pd.DataFrame(results).sort_values("p_value")
        display(anova_df.head(10))
        if self.verbose:
            print(f"✅ Global ANOVA complete. Significant features: {(anova_df['p_value'] < 0.05).sum()}")
        return anova_df

    def run_by_sector(self, df: pd.DataFrame, feature_cols: List[str], sector_col: str = "S") -> dict:
        """Run ANOVA within each sector separately."""
        sector_significant_features = {}

        for sector in df[sector_col].dropna().unique():
            sector_df = df[df[sector_col] == sector]
            significant_features = []

            for feature in feature_cols:
                pre_vals = sector_df[sector_df[self.group_col] == "pre"][feature].dropna()
                post_vals = sector_df[sector_df[self.group_col] == "post"][feature].dropna()

                if len(pre_vals) < 2 or len(post_vals) < 2:
                    continue

                # ❗ Skip if both groups are constant
                if pre_vals.nunique() <= 1 and post_vals.nunique() <= 1:
                    continue

                if len(pre_vals) > 1 and len(post_vals) > 1:
                    f_stat, p_val = f_oneway(pre_vals, post_vals)
                    if p_val < 0.05:
                        significant_features.append({
                            "feature": feature,
                            "F_statistic": f_stat,
                            "p_value": p_val
                        })

            significant_features.sort(key=lambda x: x["p_value"])
            sector_significant_features[sector] = significant_features

            # Flatten it into a list of rows
            rows = []
            for sector, feature_list in sector_significant_features.items():
                for entry in feature_list:
                    rows.append({
                        "sector": int(sector),
                        "feature": entry["feature"],
                        "F_statistic": float(entry["F_statistic"]),
                        "p_value": float(entry["p_value"]),
                    })

            # Convert to DataFrame
            anova_sector_df = pd.DataFrame(rows)

        if self.verbose:
            print(f"✅ Sector ANOVA complete. {len(sector_significant_features)} sectors analyzed.")
        return anova_sector_df, sector_significant_features

    def summarize_by_sector(self, sector_results: dict) -> pd.DataFrame:
        """Summarize how many sectors each feature was significant in."""
        feature_counts = Counter(
            feat["feature"] for features in sector_results.values() for feat in features
        )
        return pd.DataFrame(
            feature_counts.items(), columns=["feature", "num_sectors"]
        ).sort_values(by="num_sectors", ascending=False)

    def pivot_heatmap(self, sector_results: dict, df_all: pd.DataFrame, feature_map: dict) -> None:
        sector_code_to_name = df_all[["S", "sector_name"]].drop_duplicates().set_index("S")["sector_name"].to_dict()

        records = []
        for sector_code, features in sector_results.items():
            sector_name = sector_code_to_name.get(sector_code, f"Sector {int(sector_code)}")
            for feat in features:
                feat_label = feature_map.get(feat["feature"], feat["feature"])
                records.append((sector_name, feat_label))

        df_map = pd.DataFrame(records, columns=["Sector", "Feature"])
        pivot = pd.crosstab(df_map["Sector"], df_map["Feature"])

        plt.figure(figsize=(28, 12))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
        plt.title("Significant Financial Indicators per Sector (Pre vs Post COVID)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        if self.save_path:
            plt.savefig(f"{self.save_path}/anova_sector_heatmap.png")
            print(f"Heatmap saved to {self.save_path}/anova_sector_heatmap.png")
        plt.show()

    def plot_global_top_features(self, df: pd.DataFrame, top_features: List[str], description_map: Dict[str, str]) -> None:
        """Plot top global features that changed pre vs post COVID."""
        plt.figure(figsize=(14, 10))
        for i, feature in enumerate(top_features[:4], 1):
            plt.subplot(2, 2, i)
            sns.boxplot(data=df, x=self.group_col, y=feature)
            plt.title(f"{feature}: {description_map.get(feature, 'No description')}", fontsize=10)
            plt.xlabel("COVID Period")
            plt.ylabel("Value")

        plt.suptitle("Top Financial Indicators That Changed Pre vs Post COVID", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if self.save_path:
            plt.savefig(f"{self.save_path}/anova_global_top_features.png")
            print(f"✅ Global top features plot saved to {self.save_path}/anova_global_top_features.png")
        plt.show()

    def plot_by_sector_features(self, df: pd.DataFrame, sector_results: dict, feature_map: dict) -> None:
        """Boxplot of each significant feature per sector."""
        for sector_code, features in sector_results.items():
            sector_df = df[df["S"] == sector_code]
            if sector_df.empty:
                continue

            print(f"📊 Sector {int(sector_code)} — Plotting {len(features)} feature(s)")

            for feature in features:
                if feature["feature"] not in sector_df.columns:
                    continue

                plt.figure(figsize=(6, 4))
                sns.boxplot(data=sector_df, x=self.group_col, y=feature["feature"])
                description = feature_map.get(feature["feature"], "")
                plt.title(f"Sector {int(sector_code)} — {feature['feature']}: {description}")
                plt.xlabel("COVID Period")
                plt.ylabel("Value")
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                if self.save_path:
                    plt.savefig(f"{self.save_path}/anova_sector_{int(sector_code)}_{feature['feature']}.png")
                    print(f"✅ Plot saved to {self.save_path}/anova_sector_{int(sector_code)}_{feature['feature']}.png")
                plt.show()


def parse_variable_descriptions(filepath: str) -> Dict[str, str]:
    dim_df = pd.read_csv(filepath, header=None)
    dim_df = dim_df[dim_df.columns[0]].str.split(";", expand=True)
    dim_df.columns = dim_df.iloc[0]
    dim_df = dim_df[1:]  # remove header row
    dim_df["Variable Name"] = dim_df["Variable Name"].str.strip()
    return dict(zip(dim_df["Variable Name"], dim_df["Description"]))
