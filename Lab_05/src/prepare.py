"""
Stage 1: prepare
Loads the raw dataset and produces a deterministic train/test split.
California Housing regression dataset.
"""
import yaml
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data")


def load_dataset() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.rename(columns={"MedHouseVal": "label"}, inplace=True)
    return df


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    OUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"Wrote {len(train_df)} train rows, {len(test_df)} test rows")


if __name__ == "__main__":
    main()

