"""
Flask Web Application - Boston Housing Price Predictor
------------------------------------------------------
Uses the best performing model (RandomForestRegressor, R²=0.89)
to predict median house prices based on user-supplied features.

User inputs are scaled using the same StandardScaler fitted during
feature engineering, ensuring consistency with the training pipeline.

Usage:
    python app.py
    Open http://127.0.0.1:5000 in your browser
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# Paths (relative to Lab_03 root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForestRegressor.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "features", "scaler.pkl")

# ---------------------------------------------------------------------------
# Feature definitions (Boston Housing dataset)
# ---------------------------------------------------------------------------
FEATURES = [
    {"name": "CRIM",    "label": "Per Capita Crime Rate",              "default": 0.00632, "step": 0.001,  "description": "Per capita crime rate by town"},
    {"name": "ZN",      "label": "Residential Land Zoned (%)",         "default": 18.0,    "step": 0.1,    "description": "Proportion of residential land zoned for lots over 25,000 sq.ft."},
    {"name": "INDUS",   "label": "Non-Retail Business Acres (%)",      "default": 2.31,    "step": 0.01,   "description": "Proportion of non-retail business acres per town"},
    {"name": "CHAS",    "label": "Charles River (0/1)",                "default": 0,       "step": 1,      "description": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)"},
    {"name": "NOX",     "label": "Nitric Oxide Concentration (ppm)",   "default": 0.538,   "step": 0.001,  "description": "Nitric oxides concentration (parts per 10 million)"},
    {"name": "RM",      "label": "Avg Rooms per Dwelling",             "default": 6.575,   "step": 0.001,  "description": "Average number of rooms per dwelling"},
    {"name": "AGE",     "label": "Units Built Pre-1940 (%)",           "default": 65.2,    "step": 0.1,    "description": "Proportion of owner-occupied units built prior to 1940"},
    {"name": "DIS",     "label": "Distance to Employment Centers",     "default": 4.09,    "step": 0.01,   "description": "Weighted distances to five Boston employment centres"},
    {"name": "RAD",     "label": "Highway Accessibility Index",        "default": 1,       "step": 1,      "description": "Index of accessibility to radial highways"},
    {"name": "TAX",     "label": "Property Tax Rate (per $10K)",       "default": 296.0,   "step": 1,      "description": "Full-value property-tax rate per $10,000"},
    {"name": "PTRATIO", "label": "Pupil-Teacher Ratio",                "default": 15.3,    "step": 0.1,    "description": "Pupil-teacher ratio by town"},
    {"name": "B",       "label": "B Value (1000(Bk-0.63)²)",          "default": 396.9,   "step": 0.1,    "description": "1000(Bk - 0.63)² where Bk is the proportion of Black residents"},
    {"name": "LSTAT",   "label": "Lower Status Population (%)",        "default": 4.98,    "step": 0.01,   "description": "% lower status of the population"},
]

# ---------------------------------------------------------------------------
# Load model and scaler once at startup
# ---------------------------------------------------------------------------
print(f"[app] Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"[app] Loading scaler from {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_values = {}
    error = None

    if request.method == "POST":
        try:
            # Collect feature values from form
            values = []
            for feat in FEATURES:
                val = float(request.form.get(feat["name"], feat["default"]))
                input_values[feat["name"]] = val
                values.append(val)

            # Scale the input using the fitted scaler
            feature_names = [f["name"] for f in FEATURES]
            input_df = pd.DataFrame([values], columns=feature_names)
            scaled = scaler.transform(input_df)

            # Predict
            pred = model.predict(scaled)[0]
            prediction = round(float(pred), 2)

        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            # Keep submitted values for re-display
            for feat in FEATURES:
                input_values[feat["name"]] = request.form.get(feat["name"], feat["default"])
    else:
        # Pre-fill defaults on GET
        for feat in FEATURES:
            input_values[feat["name"]] = feat["default"]

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        input_values=input_values,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
