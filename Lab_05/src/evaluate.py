"""
Stage 3: evaluate
Scores the regression model on the held-out test set.
Exit code 1 stops a model below threshold from reaching the register step.
"""
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    model = joblib.load("model/model.joblib")
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    metrics = {
        "r2": r2_score(y_test, preds),
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_test, preds),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    if metrics["r2"] < params["min_r2"]:
        print(
            f"FAIL: R2 score {metrics['r2']:.4f} "
            f"is below gate {params['min_r2']}"
        )
        sys.exit(1)

    print("PASS: model cleared the quality gate")


if __name__ == "__main__":
    main()

