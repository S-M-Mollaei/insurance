import pandas as pd
import numpy as np
import pytest
from src.data_prep import DataPreparator

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "X1": list(range(15)),
        "X2": list(reversed(range(15))),
        "X3": [i * 2 for i in range(15)],
        "S": [1]*5 + [2]*5 + [3]*5  # 5 examples per class
    })

    sector_features = {
        1: [{"feature": "X1", "F_statistic": 5.0, "p_value": 0.01}],
        2: [{"feature": "X2", "F_statistic": 4.0, "p_value": 0.02}],
        3: [{"feature": "X3", "F_statistic": 6.0, "p_value": 0.03}],
    }

    return df, sector_features

def test_prepare_features(sample_data):
    df, sector_features = sample_data
    dp = DataPreparator(verbose=False)

    X_all, X_selected, y = dp.prepare_features(df, sector_features)

    # Check shapes
    assert X_all.shape == (15, 3)
    assert X_selected.shape == (15, 3)  # All 3 features used
    assert y.shape == (15,)
    assert set(X_selected.columns) == {"X1", "X2", "X3"}

def test_split(sample_data):
    df, sector_features = sample_data
    dp = DataPreparator(verbose=False)

    X_all, X_selected, y = dp.prepare_features(df, sector_features)
    splits = dp.split(X_all, X_selected, y)

    # Check structure
    assert "all" in splits and "selected" in splits

    for key in ["all", "selected"]:
        X_train, X_test, y_train, y_test = splits[key]
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) + len(y_test) == len(df)
