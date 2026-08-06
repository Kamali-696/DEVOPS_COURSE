"""
Stage 3: Feature Engineering
------------------------------
Reads the processed CSV, scales numeric features with StandardScaler,
splits into train/test sets, and writes both to data/features/.

Modification from Lab_02:
    - No logic changes; identical feature engineering pipeline
    - Reads the same params.yaml structure for test_size and random_state

Input:
    data/processed/data.csv
Output:
    data/features/train.csv
    data/features/test.csv
    data/features/scaler.pkl
"""

import os
import yaml
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(path: str = "data/processed/data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[feature_engineering] Loaded processed data (shape={df.shape})")
    return df


def build_features(df: pd.DataFrame, test_size: float, random_state: int):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.values

    test_df = X_test_scaled.copy()
    test_df["target"] = y_test.values

    return train_df, test_df, scaler


def save_features(train_df, test_df, scaler, out_dir: str = "data/features") -> None:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    scaler_path = os.path.join(out_dir, "scaler.pkl")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    print(f"[feature_engineering] Saved train -> {train_path} (shape={train_df.shape})")
    print(f"[feature_engineering] Saved test  -> {test_path} (shape={test_df.shape})")
    print(f"[feature_engineering] Saved scaler -> {scaler_path}")


def main():
    params = load_params()["feature_engineering"]
    df = load_processed_data()
    train_df, test_df, scaler = build_features(
        df, test_size=params["test_size"], random_state=params["random_state"]
    )
    save_features(train_df, test_df, scaler)


if __name__ == "__main__":
    main()
