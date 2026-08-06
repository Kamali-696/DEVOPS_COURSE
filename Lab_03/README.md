# DVC + MLflow ML Pipeline — Boston Housing Regression

This project builds an automated end-to-end Machine Learning pipeline using **DVC**, **MLflow**, and **scikit-learn** for median house-price regression on the **Boston Housing Dataset** (loaded from OpenML).

## Changes from Lab_02

| Aspect | Lab_02 | Lab_03 |
|---|---|---|
| Dataset | California Housing (sklearn) | Boston Housing (OpenML #506) |
| Models | Linear Regression only | Linear Regression, Decision Tree, Random Forest |
| Tracking | DVC metrics only | **MLflow Tracking** (params, metrics, artifacts) |
| Comparison | Single model output | Automatic best-model selection by R² score |

## 1. Install dependencies
```bash
pip install scikit-learn pandas numpy mlflow pyyaml joblib dvc
```

## 2. Initialize Git + DVC
```bash
git init
dvc init
```

## 3. Run the pipeline
```bash
dvc repro
```

## 4. View the DAG
```bash
dvc dag
```

## 5. View metrics
```bash
dvc metrics show
```

## 6. View MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open **http://127.0.0.1:5000** in your browser.

## 7. Commit to Git
```bash
git add .
git commit -m "Boston Housing regression pipeline with MLflow"
```

---

**Re-run after changing code/params:**
```bash
dvc repro
```

**Force re-run everything:**
```bash
dvc repro -f
```
