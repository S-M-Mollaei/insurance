import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class Visualizer:
    def __init__(self, sector_map: Dict[int, str], verbose: bool = True, save_path: str = None):
        self.sector_map = sector_map
        self.verbose = verbose
        self.save_path = save_path

    def plot_sector_distribution(self, df: pd.DataFrame, target_col: str = "S") -> None:
        """Plot distribution of companies by sector (bar chart)."""
        sector_counts = df[target_col].value_counts().sort_index()
        sector_names = [self.sector_map.get(s, f"Unknown ({s})") for s in sector_counts.index]

        plt.figure(figsize=(10, 5))
        plt.bar(sector_names, sector_counts.values)
        plt.title("Sector Distribution")
        plt.xlabel("Sector")
        plt.ylabel("Number of Companies")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if self.save_path:
            plt.savefig(f"{self.save_path}/sector_distribution.png")
        print("✅ Sector distribution plot generated and saved.")
        plt.show()

    def plot_pre_post_covid(self, df: pd.DataFrame, period_col: str = "period") -> None:
        """Plot sample counts pre vs post COVID."""
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=period_col, palette="Set2")
        plt.title("Samples Pre vs Post COVID")
        plt.xlabel("Period")
        plt.ylabel("Count")
        plt.tight_layout()
        if self.save_path:
            plt.savefig(f"{self.save_path}/pre_post_covid.png")
        print("✅ Pre vs Post COVID plot generated and saved.")
        plt.show()

    def plot_country_sector_heatmap(self, df: pd.DataFrame, country_col: str = "Country", sector_col: str = "sector_name") -> None:
        """Plot heatmap of companies per country × sector."""
        country_sector_counts = pd.crosstab(df[country_col], df[sector_col])

        plt.figure(figsize=(10, 6))
        sns.heatmap(country_sector_counts, annot=True, fmt="d", cmap="Reds", cbar=False)
        plt.title("Number of Companies per Country per Sector")
        plt.xlabel("Sector")
        plt.ylabel("Country")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if self.save_path:
            plt.savefig(f"{self.save_path}/country_sector_heatmap.png")
        print("✅ Country-Sector heatmap generated and saved.")
        plt.show()
