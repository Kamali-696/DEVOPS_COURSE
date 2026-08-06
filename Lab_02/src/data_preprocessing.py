"""
Stage 2: Data Preprocessing
-----------------------------
Reads the raw CSV, cleans column names, checks/handles missing values and
duplicates, and writes a processed CSV.

Input:
    data/raw/data.csv
Output:
    data/processed/data.csv
"""

import os
import pandas as pd


def load_raw_data(path: str = "data/raw/data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[data_preprocessing] Loaded raw data (shape={df.shape})")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if before != after:
        print(f"[data_preprocessing] Dropped {before - after} duplicate rows")

    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
        print("[data_preprocessing] Filled missing values with column medians")

    df["target"] = df["target"].astype(float)

    return df


def save_processed_data(df: pd.DataFrame, out_dir: str = "data/processed") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_preprocessing] Saved processed data -> {out_path} (shape={df.shape})")


def main():
    df = load_raw_data()
    df = clean_data(df)
    save_processed_data(df)


if __name__ == "__main__":
    main()