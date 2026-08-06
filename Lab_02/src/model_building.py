"""
Stage 4: Model Building
--------------------------
Trains a LinearRegression model on the engineered training features
and serializes the fitted model.

Input:
    data/features/train.csv
Output:
    model.pkl
"""

import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_data(path: str = "data/features/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[model_building] Loaded training data (shape={df.shape})")
    return df


def train_model(df: pd.DataFrame, fit_intercept: bool):
    X_train = df.drop(columns=["target"])
    y_train = df["target"]

    model = LinearRegression(
        fit_intercept=fit_intercept,
    )
    model.fit(X_train, y_train)
    print("[model_building] Model training complete")
    return model


def save_model(model, path: str = "model.pkl") -> None:
    joblib.dump(model, path)
    print(f"[model_building] Saved model -> {path}")


def main():
    params = load_params()["model_building"]
    df = load_train_data()
    model = train_model(
        df,
        fit_intercept=params.get("fit_intercept", True),
    )
    save_model(model)


if __name__ == "__main__":
    main()