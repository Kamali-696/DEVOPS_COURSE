"""
Stage 5: Model Evaluation
----------------------------
Loads all three trained models and the test set, computes regression
evaluation metrics for each, logs them to MLflow, compares performance,
and identifies the best model based on R² score.

Modification from Lab_02:
    - Lab_02 evaluated a single model and saved one metrics.json
    - Lab_03 evaluates THREE models, logs metrics to MLflow for each,
      compares R² scores, and highlights the best-performing model
    - metrics.json now contains results for all models plus a 'best_model' key

Input:
    models/LinearRegression.pkl
    models/DecisionTreeRegressor.pkl
    models/RandomForestRegressor.pkl
    data/features/test.csv
Output:
    metrics.json  (combined results for all models + best model identification)
    MLflow runs   (metrics logged for each model)
"""

import os
import json
import numpy as np  # pyrefly: ignore [missing-import]
import joblib  # pyrefly: ignore [missing-import]
import mlflow  # pyrefly: ignore [missing-import]
import pandas as pd  # pyrefly: ignore [missing-import]
from sklearn.metrics import (  # pyrefly: ignore [missing-import]
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# MLflow + DagsHub configuration (must match model_building.py)
# ---------------------------------------------------------------------------
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/kamalikamuruganandham/DEVOPS_COURSE.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "kamalikamuruganandham"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdfcb46c073aca94a1baaad365db30074bf11364"  # TODO: replace with your DagsHub token

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = "Boston_Housing_Regression"

# Models to evaluate (must match the filenames saved by model_building.py)
MODEL_NAMES = ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"]


def load_model(model_name: str, models_dir: str = "models"):
    """Load a single trained model from disk."""
    path = os.path.join(models_dir, f"{model_name}.pkl")
    model = joblib.load(path)
    print(f"[model_evaluation] Loaded model <- {path}")
    return model


def load_test_data(path: str = "data/features/test.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[model_evaluation] Loaded test data (shape={df.shape})")
    return df


def evaluate(model, df: pd.DataFrame) -> dict:
    """Compute regression metrics: R², MAE, MSE, RMSE."""
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "r2_score": float(r2),
        "mean_absolute_error": float(mae),
        "mean_squared_error": float(mse),
        "root_mean_squared_error": rmse,
    }
    return metrics


def evaluate_all_models(df: pd.DataFrame) -> dict:
    """
    Evaluate all three models, log metrics to MLflow, compare performance,
    and identify the best model based on R² score.
    """
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results = {}
    best_model_name = None
    best_r2 = -float("inf")

    for model_name in MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")

        model = load_model(model_name)
        metrics = evaluate(model, df)

        # Log metrics to MLflow under a dedicated evaluation run
        with mlflow.start_run(run_name=f"eval_{model_name}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("stage", "evaluation")
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        # Print metrics
        print(f"  R² Score : {metrics['r2_score']:.6f}")
        print(f"  MAE      : {metrics['mean_absolute_error']:.6f}")
        print(f"  MSE      : {metrics['mean_squared_error']:.6f}")
        print(f"  RMSE     : {metrics['root_mean_squared_error']:.6f}")

        # Store results
        all_results[model_name] = metrics

        # Track best model by R² score
        if metrics["r2_score"] > best_r2:
            best_r2 = metrics["r2_score"]
            best_model_name = model_name

    return all_results, best_model_name


def save_metrics(all_results: dict, best_model_name: str, path: str = "metrics.json") -> None:
    """Save combined metrics for all models and best model identification."""
    output = {
        "best_model": best_model_name,
        "best_r2_score": all_results[best_model_name]["r2_score"],
        "models": all_results,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n[model_evaluation] Saved metrics -> {path}")


def highlight_best_model(all_results: dict, best_model_name: str) -> None:
    """Print a clear summary comparing all models and highlighting the best."""
    print("\n")
    print("=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<30} {'R² Score':>10}")
    print(f"  {'-'*30} {'-'*10}")
    for name, metrics in all_results.items():
        marker = " <-- BEST" if name == best_model_name else ""
        print(f"  {name:<30} {metrics['r2_score']:>10.6f}{marker}")
    print("=" * 60)
    print(f"\n  ★ Best Model: {best_model_name}")
    print(f"    R² Score : {all_results[best_model_name]['r2_score']:.6f}")
    print(f"    MAE      : {all_results[best_model_name]['mean_absolute_error']:.6f}")
    print(f"    MSE      : {all_results[best_model_name]['mean_squared_error']:.6f}")
    print(f"    RMSE     : {all_results[best_model_name]['root_mean_squared_error']:.6f}")
    print("=" * 60)


def main():
    df = load_test_data()
    all_results, best_model_name = evaluate_all_models(df)
    save_metrics(all_results, best_model_name)
    highlight_best_model(all_results, best_model_name)


if __name__ == "__main__":
    main()
