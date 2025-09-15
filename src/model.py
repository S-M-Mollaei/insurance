# src/model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, n_estimators: int = 100, random_state: int = 42, n_splits: int = 5, verbose: bool = True, path: str = None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_splits = n_splits
        self.verbose = verbose
        self.path = path

    def cross_validate(self, X, y) -> Dict[str, float]:
        """Run StratifiedKFold cross-validation and return mean/std F1 macro."""
        clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")

        if self.verbose:
            print("✅ Cross-validation F1 Macro Scores:", scores)
            print("📊 Mean F1 Macro: {:.3f} ± {:.3f}".format(scores.mean(), scores.std()))

        return {"scores": scores, "mean": scores.mean(), "std": scores.std()}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test) -> RandomForestClassifier:
        """Train RandomForest on train data and evaluate on test data."""
        clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if self.verbose:
            print("📊 Classification Report After CV Evaluation:")
            print(classification_report(y_test, y_pred))

        return clf

    def plot_feature_importances(self, clf: RandomForestClassifier, feature_names: List[str], top_n: int = 15) -> None:
        """Plot top feature importances from a trained RandomForest model."""
        importances = pd.Series(clf.feature_importances_, index=feature_names)
        top_importances = importances.sort_values(ascending=False).head(top_n)
         
        print("🏅 Top Feature Importances Plotting:\n")

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_importances.index)
        plt.title(f"Top {top_n} Important Features for Predicting Sector")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        if self.path and len(feature_names) < 82: plt.savefig(f"{self.path}/feature_importances_selected.png")
        else: plt.savefig(f"{self.path}/feature_importances_all.png")
        plt.show()

    def tune_hyperparameters(
        self, X_train, y_train, X_test, y_test,
        param_grid: dict,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
        verbose: int = 2
    ) -> RandomForestClassifier:
        """Run GridSearchCV for hyperparameter tuning on RandomForest."""
        clf = RandomForestClassifier(random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )

        grid_search.fit(X_train, y_train)

        if self.verbose:
            print("✅ Best F1_macro score: {:.3f}".format(grid_search.best_score_))
            print("🏆 Best parameters:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        if self.verbose:
            print("📊 Classification Report (Best Model):")
            from sklearn.metrics import classification_report
            print(classification_report(y_test, y_pred))

        return best_model