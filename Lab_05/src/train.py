"""
Stage 2: train
Fits a regression model on data/train.csv and writes model/model.joblib
"""
import yaml
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = Path("model")


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    MODEL_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    reg = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    reg.fit(X_train, y_train)

    joblib.dump(reg, MODEL_DIR / "model.joblib")

    # Save feature names alongside the model, needed for HF model card / inference
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(list(X_train.columns), f)

    print("Regression model trained and saved to model/model.joblib")


if __name__ == "__main__":
    main()

