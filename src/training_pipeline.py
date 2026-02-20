import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from feast import FeatureStore
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

store = FeatureStore(repo_path="features")

entity_df = pd.DataFrame(
    {
        "student_id": range(1044),
        "event_timestamp": pd.to_datetime(
            ["2026-02-20 11:22:24.079496" for x in range(1044)]
        ),
    }
)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

mlflow.set_experiment("student-performance")

with mlflow.start_run():
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

    # Train
    pipeline.fit(X_train, y_train)
    MODEL_THRESHOLD = 0.31
    # Predict
    y_pred = pipeline.predict(X_test)
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
    print("Logged to MLFlow successfully")

    # for threshold in [0.31]:
    #     y_pred_thresh = (y_prob > threshold).astype(int)

    #     print(f"\nThreshold: {threshold}")
    #     print("Precision:", precision_score(y_test, y_pred_thresh))
    #     print("Recall:", recall_score(y_test, y_pred_thresh))
    #     print("F1:", f1_score(y_test, y_pred_thresh))
