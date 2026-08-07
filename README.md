# 🚀 DevOps Course — Lab Portfolio

> End-to-end MLOps lab exercises covering Linux scripting, DVC pipelines, MLflow experiment tracking, Docker containerisation, and CI/CD with GitHub Actions.

---

## 📂 Repository Structure

```
DEVOPS_COURSE/
├── Lab_01/   # Linux & Shell Scripting
├── Lab_02/   # DVC ML Pipeline — California Housing
├── Lab_03/   # DVC + MLflow Pipeline — Boston Housing
├── Lab_04/   # Dockerised Flask Prediction App
├── Lab_05/   # CI/CD with GitHub Actions & Hugging Face Hub
└── requirements.txt
```

---

## 🧪 Labs at a Glance

| Lab | Title | Key Tools |
|-----|-------|-----------|
| **01** | [Linux & Shell Scripting](Lab_01/) | Bash, `grep`, `awk`, environment variables |
| **02** | [DVC ML Pipeline](Lab_02/) | DVC, scikit-learn, Linear Regression |
| **03** | [DVC + MLflow Pipeline](Lab_03/) | DVC, MLflow, Multiple Models, Best-model Selection |
| **04** | [Dockerised Flask App](Lab_04/) | Docker, Flask, Model Serving |
| **05** | [CI/CD Pipeline](Lab_05/) | GitHub Actions, pytest, Hugging Face Hub |

---

## 🔬 Lab Details

### Lab 01 — Linux & Shell Scripting

Basic shell scripting fundamentals: printing system info (`whoami`, `pwd`, `date`), reading environment variables, and processing text files with student records using standard Unix utilities.

**Files:** `script.sh` · `students.txt` · `output.txt`

---

### Lab 02 — DVC ML Pipeline (California Housing)

An automated 5-stage ML pipeline built with **DVC** for the California Housing dataset:

```
data_ingestion → data_preprocessing → feature_engineering → model_building → model_evaluation
```

- **Dataset:** California Housing (scikit-learn built-in)
- **Model:** Linear Regression
- **Tracking:** DVC metrics (`metrics.json`)

**Run:** `dvc repro` · **Visualise DAG:** `dvc dag` · **View metrics:** `dvc metrics show`

---

### Lab 03 — DVC + MLflow Pipeline (Boston Housing)

Extends Lab 02 with **MLflow experiment tracking** and multi-model comparison on the Boston Housing dataset:

| Improvement | Description |
|-------------|-------------|
| Dataset | Boston Housing (OpenML #506) |
| Models | Linear Regression, Decision Tree, Random Forest |
| Tracking | MLflow Tracking (params, metrics, artifacts) |
| Selection | Automatic best-model selection by R² score |

**MLflow UI:** `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000

---

### Lab 04 — Dockerised Flask Prediction App

A **Flask** web application that serves the best-performing model (Random Forest, R² = 0.89) from Lab 03 as a REST endpoint for real-time house price predictions.

- **13 input features** with pre-filled defaults (CRIM, ZN, RM, etc.)
- **StandardScaler** applied at inference time for consistency with training
- **Containerised** with Docker (`python:3.12-slim`)

```bash
# Run locally
python app.py                    # → http://127.0.0.1:5000

# Run with Docker
docker build -t housing-app .
docker run -p 5000:5000 housing-app
```

---

### Lab 05 — CI/CD Pipeline (GitHub Actions + Hugging Face Hub)

A full CI/CD workflow that **trains, evaluates, and deploys** a model on every push to `main`:

```
push/PR → test (pytest) → train_and_evaluate → deploy_to_huggingface
```

- **Quality gate:** Deployment only proceeds if accuracy ≥ threshold in `params.yaml`
- **Auto-deploy:** Model is pushed to Hugging Face Hub on success
- **Secrets needed:** `HF_TOKEN` (GitHub Secret) · `HF_REPO_ID` (GitHub Variable)

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.12, Bash |
| **ML / Data** | scikit-learn, pandas, NumPy, joblib |
| **Pipeline** | DVC |
| **Experiment Tracking** | MLflow |
| **Web Framework** | Flask |
| **Containerisation** | Docker |
| **CI/CD** | GitHub Actions |
| **Model Registry** | Hugging Face Hub |

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/Kamali-696/DEVOPS_COURSE.git
cd DEVOPS_COURSE

# Install dependencies
pip install -r requirements.txt

# Run any lab's pipeline
cd Lab_02 && dvc repro
```

---

## 📄 License

This project is part of academic coursework (Semester V — DevOps Course).
