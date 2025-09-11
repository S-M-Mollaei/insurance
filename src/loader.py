# src/loader.py

from pathlib import Path
import pandas as pd
import numpy as np

class ARFFParser:
    @staticmethod
    def extract_year_quarter(relation: str) -> tuple[int, int]:
        relation = relation.strip().strip("'\"")
        parts = relation.split()
        try:
            if len(parts) >= 2 and parts[1].upper().startswith("Q"):
                return int(parts[0]), int(parts[1][1:])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid ARFF relation format: {relation}")

    @staticmethod
    def assign_period(year: int, quarter: int, cutoff_year: int = 2020, cutoff_quarter: int = 1) -> str:
        return "pre" if (year < cutoff_year or (year == cutoff_year and quarter <= cutoff_quarter)) else "post"


class ARFFLoader:
    def __init__(self, data_dir: Path, cutoff_year: int = 2020, cutoff_quarter: int = 1):
        self.data_dir = data_dir
        self.cutoff_year = cutoff_year
        self.cutoff_quarter = cutoff_quarter

    def _check_violation(self, df: pd.DataFrame, attr_df: pd.DataFrame, file: Path):
        for _, row in attr_df.iterrows():
            name = row["name"]
            meta = row["type_raw"]
            allowed_set = {str(t).strip() for t in meta}

            if name != 'S':
                s = df[name].astype("string").str.strip("'")
                mask_bad = ~s.isin(allowed_set)
            else:
                s = pd.to_numeric(df["S"], errors="coerce")
                mask_bad = df["S"].notna() & s.isna()

            if mask_bad.any():
                print(f"❌ Violations in {file.name} for '{name}': {s[mask_bad].value_counts().to_dict()}")

    def load_file(self, filepath: Path) -> pd.DataFrame:
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Parse attribute metadata
        attr_lines = [line for line in lines if line.lower().startswith("@attribute")]
        attr_names, attr_types = [], []

        for line in attr_lines:
            parts = line.strip().split(maxsplit=2)
            if len(parts) < 3:
                continue
            attr_names.append(parts[1])
            type_str = parts[2].strip()
            if type_str.startswith("{") and type_str.endswith("}"):
                values = [x.strip().strip("'").strip('"') for x in type_str[1:-1].split(",")]
                attr_types.append(values)
            else:
                attr_types.append(type_str)

        attr_df = pd.DataFrame({"name": attr_names, "type_raw": attr_types})

        # Read data section
        data_start = next(i for i, line in enumerate(lines) if line.lower().strip() == "@data")
        data_lines = lines[data_start + 1:]
        data = [line.strip().split(",") for line in data_lines if line.strip() and not line.startswith("%")]

        df = pd.DataFrame(data, columns=attr_names)

        print(f"🔍 Checking violation for: {filepath.name}")
        self._check_violation(df, attr_df, filepath)

        # Replace missing values
        df = df.replace(r"^\s*[mM]\s*$", np.nan, regex=True)

        # Convert numeric
        feature_cols = [col for col in df.columns if col.startswith("X")]
        df[feature_cols + ["S"]] = df[feature_cols + ["S"]].apply(pd.to_numeric, errors="coerce")
        df["S"] = pd.to_numeric(df["S"], errors="coerce").astype("Int64")

        # Metadata
        relation_line = next((line for line in lines if line.lower().startswith("@relation")), None)
        relation_str = " ".join(relation_line.split()[1:]) if relation_line else ""
        year, quarter = ARFFParser.extract_year_quarter(relation_str)
        df["year"] = year
        df["quarter"] = quarter
        df["period"] = df.apply(lambda row: ARFFParser.assign_period(year, quarter, self.cutoff_year, self.cutoff_quarter), axis=1)

        return df

    def load_all(self) -> pd.DataFrame:
        arff_files = sorted(self.data_dir.glob("*.arff"))
        df_list = [self.load_file(file) for file in arff_files]
        df_all = pd.concat(df_list, ignore_index=True)

        # Clean: Drop all-zero or all-NaN rows in features
        feature_cols = [col for col in df_all.columns if col.startswith("X")]
        mask_all_zero_or_nan = df_all[feature_cols].fillna(0).eq(0).all(axis=1)
        df_all = df_all.loc[~mask_all_zero_or_nan].copy()

        # Fill remaining NaNs with median
        df_all[feature_cols] = df_all[feature_cols].fillna(df_all[feature_cols].median())

        print("✅ All ARFF files loaded and stacked.")
        return df_all
