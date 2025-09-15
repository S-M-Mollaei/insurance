import pandas as pd
from src.anova import ANOVAAnalyzer

def test_run_by_sector_returns_expected():
    # Create test data
    df = pd.DataFrame({
        'X1': [1, 2, 3, 10, 11, 12, 20, 21, 22],
        'period': ['pre', 'pre', 'pre', 'post', 'post', 'post', 'pre', 'pre', 'post'],
        'S': [1, 1, 1, 1, 1, 1, 2, 2, 2]
    })

    # Run sector-wise ANOVA
    analyzer = ANOVAAnalyzer(group_col="period", verbose=False)
    df_anova, sector_dict = analyzer.run_by_sector(df, feature_cols=["X1"], sector_col="S")

    # Assertions
    assert isinstance(df_anova, pd.DataFrame)
    assert "sector" in df_anova.columns
    assert "feature" in df_anova.columns
    assert "p_value" in df_anova.columns
    assert isinstance(sector_dict, dict)
    assert all(isinstance(v, list) for v in sector_dict.values())

def test_summarize_by_sector_counts_features():
    # Simulated output from run_by_sector()
    dummy_sector_dict = {
        1.0: [{"feature": "X1", "F_statistic": 4.2, "p_value": 0.03}],
        2.0: [{"feature": "X1", "F_statistic": 6.5, "p_value": 0.01},
              {"feature": "X2", "F_statistic": 5.0, "p_value": 0.04}]
    }

    analyzer = ANOVAAnalyzer(verbose=False)
    summary_df = analyzer.summarize_by_sector(dummy_sector_dict)

    # Assertions
    assert isinstance(summary_df, pd.DataFrame)
    assert "feature" in summary_df.columns
    assert "num_sectors" in summary_df.columns
    assert summary_df.loc[summary_df['feature'] == 'X1', 'num_sectors'].values[0] == 2
    assert summary_df.loc[summary_df['feature'] == 'X2', 'num_sectors'].values[0] == 1