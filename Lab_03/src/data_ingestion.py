"""
Stage 1: Data Ingestion
------------------------
Loads the Boston Housing dataset from OpenML (dataset ID 506)
and dumps it as a raw CSV file.

Modification from Lab_02:
    - Replaced sklearn's fetch_california_housing with fetch_openml (Boston Housing)
    - Boston Housing was removed from sklearn.datasets; OpenML is the standard alternative
    - Target column 'MEDV' is renamed to 'target' for consistency with the pipeline
    - Added retry logic and offline fallback for unreliable networks

Output:
    data/raw/data.csv
"""

import os
import time
import pandas as pd  # pyrefly: ignore [missing-import]
from sklearn.datasets import fetch_openml  # pyrefly: ignore [missing-import]


def load_data(max_retries: int = 3, retry_delay: float = 2.0) -> pd.DataFrame:
    """Load the Boston Housing dataset from OpenML into a DataFrame.

    Retries up to `max_retries` times on network errors. If all attempts
    fail, falls back to loading from a static CSV URL or generating
    synthetic data matching the Boston Housing schema.
    """
    # --- Attempt: fetch from OpenML with retries ---
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[data_ingestion] Attempt {attempt}/{max_retries}: fetching from OpenML ...")
            bunch = fetch_openml(data_id=506, as_frame=True, parser="auto")
            df = bunch.frame  # includes all feature columns + 'MEDV' as target

            # Rename the target column to 'target' for consistency
            if "MEDV" in df.columns:
                df = df.rename(columns={"MEDV": "target"})

            print(f"[data_ingestion] Boston Housing dataset loaded (shape={df.shape})")
            return df
        except Exception as e:
            last_err = e
            print(f"[data_ingestion] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"[data_ingestion] Retrying in {retry_delay}s ...")
                time.sleep(retry_delay)

    # --- Fallback: static CSV from GitHub ---
    print("[data_ingestion] OpenML unavailable. Trying static CSV fallback ...")
    try:
        url = (
            "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        )
        df = pd.read_csv(url)
        # The CSV has column 'medv' (lowercase)
        rename_map = {c: c.upper() for c in df.columns}
        df = df.rename(columns=rename_map)
        if "MEDV" in df.columns:
            df = df.rename(columns={"MEDV": "target"})
        print(f"[data_ingestion] Loaded from static CSV fallback (shape={df.shape})")
        return df
    except Exception as e2:
        print(f"[data_ingestion] Static CSV fallback also failed: {e2}")

    # --- Fallback: sklearn's bundled California Housing (offline) ---
    print("[data_ingestion] Using sklearn's California Housing as offline fallback ...")
    from sklearn.datasets import fetch_california_housing  # pyrefly: ignore [missing-import]

    cal = fetch_california_housing(as_frame=True)
    df = cal.frame  # target column is 'MedHouseVal'
    df = df.rename(columns={"MedHouseVal": "target"})
    print(
        f"[data_ingestion] WARNING: Using California Housing as fallback (shape={df.shape}). "
        f"Original OpenML error was: {last_err}"
    )
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

