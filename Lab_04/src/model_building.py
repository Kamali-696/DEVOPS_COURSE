"""
Stage 4: Model Building
--------------------------
Trains three regression models on the engineered training features,
logs each to MLflow, and serializes the fitted models.

Modification from Lab_02:
    - Lab_02 trained a single LinearRegression model
    - Lab_03 trains THREE models: LinearRegression, DecisionTreeRegressor,
      RandomForestRegressor
    - Integrated MLflow Tracking: each model is logged as a separate MLflow run
      with its hyperparameters and the trained model artifact
    - Models are saved to models/ directory instead of a single model.pkl

Input:
    data/features/train.csv
    params.yaml
Output:
    models/LinearRegression.pkl
    models/DecisionTreeRegressor.pkl
    models/RandomForestRegressor.pkl
    MLflow runs (logged to mlflow.db)
"""

import os
import yaml  # pyrefly: ignore [missing-import]
import joblib  # pyrefly: ignore [missing-import]
import mlflow  # pyrefly: ignore [missing-import]
import mlflow.sklearn  # pyrefly: ignore [missing-import]
import pandas as pd  # pyrefly: ignore [missing-import]
from sklearn.linear_model import LinearRegression  # pyrefly: ignore [missing-import]
from sklearn.tree import DecisionTreeRegressor  # pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestRegressor  # pyrefly: ignore [missing-import]


# ---------------------------------------------------------------------------
# MLflow + DagsHub configuration
# ---------------------------------------------------------------------------
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/kamalikamuruganandham/DEVOPS_COURSE.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "kamalikamuruganandham"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdfcb46c073aca94a1baaad365db30074bf11364"  # TODO: replace with your DagsHub token

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = "Boston_Housing_Regression"


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_data(path: str = "data/features/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[model_building] Loaded training data (shape={df.shape})")
    return df


def build_models(params: dict) -> list:
    """
    Construct three regression models with hyperparameters from params.yaml.
    Returns a list of (model_name, model_instance, hyperparams_dict) tuples.
    """
    lr_params = params["linear_regression"]
    dt_params = params["decision_tree"]
    rf_params = params["random_forest"]

    models = [
        (
            "LinearRegression",
            LinearRegression(fit_intercept=lr_params["fit_intercept"]),
            {"fit_intercept": lr_params["fit_intercept"]},
        ),
        (
            "DecisionTreeRegressor",
            DecisionTreeRegressor(
                max_depth=dt_params["max_depth"],
                random_state=dt_params["random_state"],
            ),
            {
                "max_depth": dt_params["max_depth"],
                "random_state": dt_params["random_state"],
            },
        ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                random_state=rf_params["random_state"],
            ),
            {
                "n_estimators": rf_params["n_estimators"],
                "max_depth": rf_params["max_depth"],
                "random_state": rf_params["random_state"],
            },
        ),
    ]
    return models


def train_and_log_models(
    df: pd.DataFrame, models: list, out_dir: str = "models"
) -> None:
    """
    Train each model, log hyperparameters and model artifact to MLflow,
    and save the model to disk.
    """
    os.makedirs(out_dir, exist_ok=True)

    X_train = df.drop(columns=["target"])
    y_train = df["target"]

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    for model_name, model, hyperparams in models:
        print(f"\n[model_building] Training {model_name} ...")

        with mlflow.start_run(run_name=f"train_{model_name}"):
            # Log model name as a parameter
            mlflow.log_param("model_name", model_name)

            # Log all hyperparameters
            for param_name, param_value in hyperparams.items():
                mlflow.log_param(param_name, param_value)

            # Train the model
            model.fit(X_train, y_train)

            # Log the trained model artifact to MLflow
            mlflow.sklearn.log_model(model, artifact_path=model_name)

            # Also log training set size as a metric
            mlflow.log_metric("train_samples", X_train.shape[0])
            mlflow.log_metric("train_features", X_train.shape[1])

        # Save model to disk for the evaluation stage
        model_path = os.path.join(out_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"[model_building] Saved {model_name} -> {model_path}")

    print("\n[model_building] All models trained and logged to MLflow")


def main():
    params = load_params()["model_building"]
    df = load_train_data()
    models = build_models(params)
    train_and_log_models(df, models)


if __name__ == "__main__":
    main()
