# Dataset

This project uses the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance). This dataset contains student achievement data in secondary education of two Portuguese schools. The data includes student grades, demographic, social, and school-related features, and is commonly used for predictive modeling tasks such as dropout prediction, grade prediction, and more.

Key features include:
- Demographics (age, gender, family background)
- Academic performance (grades, failures)
- Attendance and study time
- Social and family factors

For more details, see the [UCI dataset page](https://archive.ics.uci.edu/dataset/320/student+performance).

## Reference Kaggle Notebook

For a detailed exploratory analysis and baseline model, see my Kaggle notebook: [UCI Student Dropout Prediction Model](https://www.kaggle.com/code/aravindgshanthi/uci-student-dropout-prediction-model)

# DVC Pipeline Architecture

This project uses DVC (Data Version Control) to orchestrate and version the ML pipeline steps. The pipeline is defined in `dvc.yaml` and consists of the following stages:

| Stage              | Purpose                                      | Command                        | Inputs/Dependencies                        | Outputs                                    |
|--------------------|----------------------------------------------|--------------------------------|--------------------------------------------|--------------------------------------------|
| etl                | Data extraction, transformation, and loading | `python src/etl.py`            | `src/etl.py`, `dataset/`                   | `data/processed/labels.parquet`, `data/processed/student_features.parquet` |
| feast_setup        | Initialize Feast feature store registry      | `python src/feast_setup.py`    | `src/feast_setup.py`                       | `features/data/registry.db`                |
| feast_materialize  | Materialize features for training/scoring    | `python src/materialize.py`    | `data/processed/student_features.parquet`, `src/materialize.py` | `data/feast_materialized.marker`           |
| drift              | Data drift detection and auto-retraining     | `python src/drift.py`          | `data/processed/student_features.parquet`, `src/drift.py`       | `metrics/drift.json`                       |

**How it works:**

- Each stage specifies its command (`cmd`), dependencies (`deps`), and outputs (`outs`).
- DVC tracks changes in dependencies and only re-runs stages when inputs change.
- Outputs are versioned and can be pushed/pulled from remote storage (e.g., S3).
- The pipeline can be reproduced end-to-end with `dvc repro`, ensuring reproducibility and traceability.

This modular approach makes it easy to manage data, code, and model lineage across the ML workflow.

# End-to-End ML Pipeline: Student Dropout Prediction

This project implements a complete end-to-end machine learning pipeline to predict student dropout. It covers data ingestion, feature engineering, model training, experiment tracking, drift detection, and automated retraining.

## Technologies Used

- Python
- Feast (feature store) with Redis
- DVC (Data Version Control) with S3 storage
- MLflow (experiment tracking and model registry)
- Docker (containerization)
- GitHub Workflows (CI/CD automation)
- Automated model training
- Data drift detection (Population Stability Index, PSI)

## Overview

The pipeline includes:
- Data ingestion and processing
- Feature engineering and storage with Feast
- Versioned data and pipeline steps using DVC
- Model training and evaluation
- Experiment tracking and model registry with MLflow
- Drift detection using PSI and automated retraining
- Containerized deployment with Docker
- Automated workflows using GitHub Actions

## Environment Variables

Set the following environment variables before running the pipeline (for DVC S3, MLflow, etc.):

```sh
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=ap-south-1
# For MLflow S3 artifact store (if used)
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
# For Redis (if using password)
export REDIS_PASSWORD=your-redis-password
# For PostgreSQL MLflow backend (optional)
export MLFLOW_BACKEND_STORE_URI=postgresql://user:password@host:port/dbname
# For MLflow tracking URI (if running in Docker)
export MLFLOW_TRACKING_URI=http://mlflow:5001
```

## Startup Commands

Typical workflow:

```sh
# 1. Run ETL pipeline (when new data arrives)
dvc repro etl

# 2. Apply Feast feature definitions (when features change)
feast apply

# 3. Materialize new features
python src/materialize.py

# 4. Run drift detection (scheduled or manual)
python src/drift.py

# 5. Push data and models to remote storage
dvc push

# 6. Start FastAPI service (for inference)
uvicorn app:app --reload

# 7. Start MLflow server (if not using Docker Compose)
mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --default-artifact-root s3://your-bucket/mlruns --host 0.0.0.0 --port 5001
```


Remove local sync up (DVC)
dvc gc -w -c