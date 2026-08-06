"""
Stage 1: Data Ingestion
------------------------
Loads the scikit-learn California Housing dataset (regression: median house value)
and dumps it as a raw CSV file.

Output:
    data/raw/data.csv
"""

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data() -> pd.DataFrame:
    """Load the sklearn California Housing dataset into a DataFrame."""
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame  # includes feature columns + 'MedHouseVal' as target
    if "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": "target"})
    return df


def save_raw_data(df: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_ingestion] Saved raw data -> {out_path} (shape={df.shape})")


def main():
    df = load_data()
    save_raw_data(df)


if __name__ == "__main__":
    main()