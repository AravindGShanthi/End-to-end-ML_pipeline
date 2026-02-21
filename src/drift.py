import json
import os
import subprocess
import tempfile

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://mlflow:5001"
MODEL_NAME = "student_dropout_model"
CURRENT_FEATURE_PATH = "data/processed/student_features.parquet"
DRIFT_THRESHOLD = 0.2
METRICS_PATH = "metrics/drift.json"
FEATURE_REPO_PATH = "features"

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"


def get_reference_commit():
    try:
        print("get_reference_commit => ", flush=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print("get_reference_commit mlflow => ", flush=True)
        client = MlflowClient()
        print("get_reference_commit client => ", flush=True)

        model_version = client.get_model_version_by_alias(
            name=MODEL_NAME, alias="champion"
        )
        print("get_reference_commit =>model_version ", model_version, flush=True)
        run_id = model_version.run_id

        print(run_id, flush=True)

        run = mlflow.get_run(run_id)
        reference_commit = run.data.params["git_commit"]

        print("==> get_reference_pd", flush=True)
        print("COmmit => ", reference_commit, flush=True)
        return reference_commit
    except Exception as e:
        print("Registered model not found. Skipping drift", e, flush=True)
        return None


def load_refernce_dataset(reference_commit):
    print("load_refernce_dataset => ", reference_commit, flush=True)
    temp_dir = tempfile.mkdtemp()

    subprocess.run(["git", "clone", ".", temp_dir], check=True)

    subprocess.run(["git", "checkout", reference_commit], cwd=temp_dir, check=True)

    subprocess.run(["dvc", "pull"], cwd=temp_dir, check=True)

    path = os.path.join(temp_dir, "data/processed/student_features.parquet")

    return pd.read_parquet(path)


def get_reference_pd():
    print("==> get_reference_pd", flush=True)
    reference_commit = get_reference_commit()
    if reference_commit is None:
        return None
    return load_refernce_dataset(reference_commit)


# Population Density Index
def calculate_psi(expected, actual, bins=10):
    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi_values = (expected_percents - actual_percents) * np.log(
        (expected_percents + 1e-6) / (actual_percents + 1e-6)
    )

    return np.sum(psi_values)


def main():
    print("Drift started==>", flush=True)
    os.makedirs("metrics", exist_ok=True)

    # store = FeatureStore(repo_path=FEATURE_REPO_PATH)
    # store.materialize_incremental(end_date=pd.Timestamp.now())

    reference_df = get_reference_pd()

    if reference_df is None:
        print("Drift not happening", flush=True)
        from training_pipeline import train_with_auto_threshold

        train_with_auto_threshold()
        return

    current_df = pd.read_parquet(CURRENT_FEATURE_PATH)

    numeric_cols = reference_df.select_dtypes(include=np.number).columns
    current_numcols = current_df.select_dtypes(include=np.number).columns

    print("-" * 60)
    print(numeric_cols)
    print(current_numcols)
    print("-" * 60)

    drift_scores = {}
    drift_detected = False

    for col in numeric_cols:
        psi_score = calculate_psi(reference_df[col], current_df[col])
        drift_scores[col] = float(psi_score)

        if psi_score > DRIFT_THRESHOLD:
            drift_detected = True

    result = {
        "drift_detected": drift_detected,
        "psi_threshold": DRIFT_THRESHOLD,
        "feature_psi": drift_scores,
    }

    if not drift_detected:
        print("No training needed")
    else:
        print("Re-training")
        from training_pipeline import train_with_auto_threshold

        train_with_auto_threshold()

    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("Drift result:", result)


if __name__ == "__main__":
    main()
