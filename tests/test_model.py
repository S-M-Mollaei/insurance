import pytest
import pandas as pd
import numpy as np
from src.model import ModelTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from unittest.mock import patch

@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_redundant=0,
        random_state=42
    )
    columns = [f"X{i}" for i in range(1, 11)]
    df = pd.DataFrame(X, columns=columns)
    series = pd.Series(y)
    return df, series

def test_cross_validate(synthetic_data):
    X, y = synthetic_data
    trainer = ModelTrainer(verbose=False)
    result = trainer.cross_validate(X, y)

    assert "scores" in result
    assert "mean" in result
    assert "std" in result
    assert isinstance(result["mean"], float)

def test_train_and_evaluate(synthetic_data):
    X, y = synthetic_data
    trainer = ModelTrainer(verbose=False)
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    clf = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    assert isinstance(clf, RandomForestClassifier)
    assert hasattr(clf, "feature_importances_")

@patch("matplotlib.pyplot.savefig")
def test_plot_feature_importances(mock_savefig, synthetic_data):
    X, y = synthetic_data
    trainer = ModelTrainer(verbose=False)
    clf = RandomForestClassifier().fit(X, y)

    trainer.plot_feature_importances(clf, X.columns.tolist(), top_n=5)
    mock_savefig.assert_called()  # Confirm no crash and plotting works

def test_tune_hyperparameters(synthetic_data):
    X, y = synthetic_data
    trainer = ModelTrainer(verbose=False, n_splits=3)

    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    param_grid = {
        "n_estimators": [10],  # small grid to keep test fast
        "max_depth": [None]
    }

    best_model = trainer.tune_hyperparameters(X_train, y_train, X_test, y_test, param_grid, verbose=0)
    assert isinstance(best_model, RandomForestClassifier)
