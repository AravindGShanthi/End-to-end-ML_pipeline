import os
import subprocess

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from feast import FeatureStore
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
store = FeatureStore(repo_path="features")

offline_df = pd.read_parquet("data/processed/student_features.parquet")

entity_df = offline_df[["student_id", "event_timestamp"]]

X = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "student_features:fail_abs",
        "student_features:G1",
        "student_features:failures",
        "student_features:abs_log",
        "student_features:studytime",
        "student_features:Medu",
        "student_features:Fedu",
        "student_features:age",
    ],
).to_df()

y_df = pd.read_parquet("data/processed/labels.parquet")
y_df["event_timestamp"] = pd.to_datetime(y_df["event_timestamp"], utc=True)

print(y_df, X)

traning_df = X.merge(y_df, on=["student_id", "event_timestamp"])

print("traning_df", traning_df)

X = traning_df.drop(columns=["dropout", "student_id", "event_timestamp"])
Y = traning_df["dropout"]


client = MlflowClient()

exp = client.get_experiment_by_name("student-performance")

if exp and exp.lifecycle_stage == "deleted":
    client.restore_experiment(exp.experiment_id)

mlflow.set_experiment("student-performance")


def find_stable_threshold(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    thresholds = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        y_prob = model_clone.predict_proba(X_val)[:, 1]

        precision, recall, thr = precision_recall_curve(y_val, y_prob)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        print("--" * 100)
        print(f1_scores)

        best_thr = thr[np.argmax(f1_scores[:-1])]
        thresholds.append(best_thr)

    return np.median(thresholds), np.std(thresholds)


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def commit_codeRepo(commit_msg="Auto commit: Retrain"):
    try:
        subprocess.run(
            ["git", "config", "--global", "user.name", os.getenv("GIT_USER_NAME")]
        )
        subprocess.run(
            ["git", "config", "--global", "user.email", os.getenv("GIT_USER_EMAIL")]
        )

        github_token = os.getenv("GIT_TOKEN")
        github_repo = os.getenv("GITHUB_REPO_URL")  # e.g., ://github.com

        print(github_repo, github_token, flush=True)

        if github_token and github_repo:
            authenticated_url = (
                f"https://{os.getenv('GIT_USER_NAME')}:{github_token}@{github_repo}"
            )
            print("AUTH => ", authenticated_url, flush=True)
            subprocess.run(
                ["git", "remote", "set-url", "origin", authenticated_url], check=True
            )

        print("Staging git changes...", flush=True)
        subprocess.run(["git", "add", "."], cwd="/app", check=True)

        print("Commiting git changes...", flush=True)
        subprocess.run(["git", "commit", "-m", commit_msg], cwd="/app", check=True)

        print("Pushing git changes...", flush=True)
        subprocess.run(["git", "push"], cwd="/app", check=True)
    except subprocess.CalledProcessError as e:
        print(e, flush=True)
        print("Error commiting_codeRepo => ", flush=True)


def detect_threshold_drift(model_name, new_threshold, drift_limit=0.05):
    try:
        client = mlflow.tracking.MlflowClient()

        lastest_version = client.get_latest_versions(
            model_name,
        )

        if not lastest_version:
            print("No Previous production model found")
            return None, False

        lastest_run_id = lastest_version[0].run_id

        previous_threshold = float(
            client.get_run(lastest_run_id).data.params.get("decision_threshold", 0.5)
        )

        drift = abs(new_threshold - previous_threshold)

        drift_detected = drift > drift_limit

        print(f"Previous Threshold: {previous_threshold}")
        print(f"New Threshold: {new_threshold}")
        print(f"Threshold Drift: {drift}")
        print(f"Drift Detected (> {drift_limit}): {drift_detected}")

        return drift, drift_detected
    except Exception as e:
        print("Registered model not found. Skipping drift", e, flush=True)
        return None, None


def train_with_auto_threshold():
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000), method="isotonic", cv=5
                ),
            ),
        ]
    )

    best_threshold, threshold_std = find_stable_threshold(pipeline, X_train, y_train)

    print("--" * 100)
    print("Best threshold => ", best_threshold)
    print("Best threshold_std => ", threshold_std)
    print("--" * 100)

    # Train
    pipeline.fit(X_train, y_train)
    MODEL_THRESHOLD = best_threshold
    # Predict
    # y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > MODEL_THRESHOLD).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("accuracy", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    print("roc_auc", auc)

    drift, drift_flag = detect_threshold_drift("student_dropout_model", best_threshold)

    with mlflow.start_run():
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log parametrics
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("calibraction", "isotonic")

        # Log model
        mlflow.sklearn.log_model(
            pipeline, "model", registered_model_name="student_dropout_model"
        )

        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.savefig("confustion_matrix.png")

        mlflow.log_artifact("confustion_matrix.png")
        mlflow.log_param("decision_threshold", MODEL_THRESHOLD)
        commit_codeRepo("Auto update dvc.lock and metrics after retraining")
        git_commit = get_git_commit()
        mlflow.log_param("git_commit", git_commit)
        print("Logged to MLFlow successfully")
        feature_order = X_train.columns.tolist()
        mlflow.log_param("feature_order", feature_order)

        if drift is not None:
            mlflow.log_metric("threshold_drift", drift)
            mlflow.log_metric("threshold_drift_flag", drift_flag)

    lastest_version = client.get_latest_versions("student_dropout_model")[0].version

    client.set_registered_model_alias(
        name="student_dropout_model", alias="champion", version=lastest_version
    )


if __name__ == "__main__":
    train_with_auto_threshold()
