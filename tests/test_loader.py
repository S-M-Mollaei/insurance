# tests/test_loader.py

import tempfile
from pathlib import Path
import pandas as pd
from src.loader import ARFFLoader, ARFFParser

# --- Tests for ARFFParser ---

def test_extract_year_quarter_valid():
    assert ARFFParser.extract_year_quarter("2019 Q3") == (2019, 3)
    assert ARFFParser.extract_year_quarter("'2020 Q1'") == (2020, 1)

def test_extract_year_quarter_invalid():
    try:
        ARFFParser.extract_year_quarter("InvalidName")
        assert False, "Expected ValueError"
    except ValueError:
        assert True

def test_assign_period_logic():
    assert ARFFParser.assign_period(2019, 4) == "pre"
    assert ARFFParser.assign_period(2020, 2) == "post"

# --- Test for ARFFLoader.load_file ---

def test_load_file_with_minimal_arff():
    arff_content = """
@relation '2020 Q1'
@attribute X1 numeric
@attribute X2 numeric
@attribute S {1,2,3}
@data
1.0,2.0,1
3.0,4.0,2
?,6.0,3
"""

    with tempfile.NamedTemporaryFile("w+", suffix=".arff", delete=False) as tmp:
        tmp.write(arff_content)
        tmp_path = Path(tmp.name)

    loader = ARFFLoader(data_dir=tmp_path.parent)
    df = loader.load_file(tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert set(["X1", "X2", "S", "year", "quarter", "period"]).issubset(df.columns)
    assert df["period"].iloc[0] == "pre"
    assert pd.isna(df["X1"].iloc[2])
