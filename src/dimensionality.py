import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAReducer:
    def __init__(self, n_components: int = 2, feature_prefix: str = "X", verbose: bool = True, save_path: str = None):
        self.n_components = n_components
        self.feature_prefix = feature_prefix
        self.verbose = verbose
        self.pca = None
        self.save_path = save_path

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA and append principal components to the DataFrame."""
        feature_cols = [col for col in df.columns if col.startswith(self.feature_prefix)]
        X = df[feature_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.pca = PCA(n_components=self.n_components)
        components = self.pca.fit_transform(X_scaled)

        for i in range(self.n_components):
            df[f"PC{i+1}"] = components[:, i]

        if self.verbose:
            print("Explained variance ratio:", self.pca.explained_variance_ratio_)

        return df

    def plot(self, df: pd.DataFrame, hue: str = "sector_name", figsize: tuple = (10, 6), definition: str = None) -> None:
        """Scatterplot of first two PCA components colored by a categorical column."""
        if "PC1" not in df.columns or "PC2" not in df.columns:
            raise ValueError("DataFrame must contain PC1 and PC2 columns. Run fit_transform() first.")

        plt.figure(figsize=figsize)
        sns.scatterplot(
            data=df, x="PC1", y="PC2",
            hue=hue, palette="Set2", s=60
        )
        plt.title("Definition" if definition is None else f"PCA Plot - {definition}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        
        if self.save_path and definition:
            plt.savefig(f"{self.save_path}/pca_{definition.replace(' ', '_').lower()}.png")
        print("✅ PCA plot generated and saved.")

    def filter_outliers(self, df: pd.DataFrame, method: str = "iqr", limits: tuple = (-50, 50)) -> pd.DataFrame:
        """Filter out extreme points in PCA space (IQR or manual limits)."""
        if method == "manual":
            return df[(df["PC1"].between(*limits)) & (df["PC2"].between(*limits))]

        elif method == "iqr":
            Q1 = df[["PC1", "PC2"]].quantile(0.25)
            Q3 = df[["PC1", "PC2"]].quantile(0.75)
            IQR = Q3 - Q1
            bounds = ((df[["PC1", "PC2"]] >= (Q1 - 1.5 * IQR)) &
                      (df[["PC1", "PC2"]] <= (Q3 + 1.5 * IQR)))
            return df[bounds.all(axis=1)]

        else:
            raise ValueError("method must be 'manual' or 'iqr'")
