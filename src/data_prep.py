import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List

class DataPreparator:
    def __init__(self, test_size: float = 0.2, random_state: int = 42, verbose: bool = True):
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

    def prepare_features(
        self, df: pd.DataFrame, sector_significant_features: Dict[int, List[dict]], target_col: str = "S"
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepare X_all, X_selected, and y."""
        # All features
        X_all = df[[col for col in df.columns if col.startswith("X")]]
        
        # Features selected from ANOVA (by sector)
        significant_features = set()
        for feats in sector_significant_features.values():
            for feat in feats:
                significant_features.add(feat["feature"])

        X_selected = df[list(significant_features)] if significant_features else X_all.copy()
        y = df[target_col]

        if self.verbose:
            print(f"✅ Prepared features: X_all={X_all.shape}, X_selected={X_selected.shape}, y={y.shape}")

        return X_all, X_selected, y

    def split(
        self, X_all: pd.DataFrame, X_selected: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Split into train/test sets for both all and selected features."""
        splits = {}

        # All features
        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
            X_all, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        splits["all"] = (X_train_all, X_test_all, y_train_all, y_test_all)

        # Selected features
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
            X_selected, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        splits["selected"] = (X_train_sel, X_test_sel, y_train_sel, y_test_sel)

        if self.verbose:
            print("✅ Data split complete")
            for k, (Xtr, Xte, _, _) in splits.items():
                print(f"   - {k}: train={Xtr.shape}, test={Xte.shape}")

        return splits
