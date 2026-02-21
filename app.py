import os
import subprocess
import threading
import time
from datetime import datetime

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

FEAST_REPO_PATH = "features"
MODEL_URI = "models:/student_dropout_model@champion"

store = FeatureStore(repo_path=FEAST_REPO_PATH)
app = FastAPI(title="student dropout prediction API")

model = None
model_lock = threading.Lock()


def load_model():
    global model
    try:
        print("Loading model from MLflow...", flush=True)
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Model loaded successfully.", flush=True)
    except Exception as e:
        print("Error loading model ", e)


def get_model():
    global model
    with model_lock:
        if model is None:
            load_model()
        return model


class StudentRequest(BaseModel):
    student_id: int


@app.post("/predict")
def predict(request: StudentRequest):
    entity_df = pd.DataFrame(
        {
            "student_id": [request.student_id],
            "event_timestamp": [datetime.utcnow()],
        }
    )

    feature_vector = store.get_online_features(
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
        entity_rows=entity_df.to_dict(orient="records"),
    ).to_df()

    EXPECTED_FEATURES = [
        "fail_abs",
        "G1",
        "failures",
        "abs_log",
        "studytime",
        "Medu",
        "Fedu",
        "age",
    ]

    feature_vector = feature_vector[EXPECTED_FEATURES]

    print("feature_vector", flush=True)
    print(feature_vector, flush=True)

    model_final = get_model()
    try:
        model_final = get_model()
        prediction = model_final.predict(feature_vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "student_id": request.student_id,
        "prediction": float(prediction[0]),
    }


def commit_codeRepo(commit_msg="Auto commit: Retrain"):
    try:
        subprocess.run(
            ["git", "config", "--global", "user.name", os.getenv("GIT_USER_NAME")]
        )
        subprocess.run(
            ["git", "config", "--global", "user.email", os.getenv("GIT_USER_EMAIL")]
        )

        github_token = os.getenv("GITHUB_TOKEN")
        github_repo = os.getenv("GITHUB_REPO_URL")  # e.g., ://github.com

        if github_token and github_repo:
            authenticated_url = (
                f"https://{os.getenv('GIT_USER_NAME')}:{github_token}@{github_repo}"
            )
            print("AUTH => ", authenticated_url, flush=True)
            subprocess.run(
                ["git", "remote", "set-url", "origin", authenticated_url], check=True
            )

        print("Staging DVC changes...", flush=True)
        subprocess.run(["dvc", "add", "."], check=False)

        print("Staging git changes...", flush=True)
        subprocess.run(["git", "add", "."], check=True)

        print("Commiting git changes...", flush=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)

        print("Pushing git changes...", flush=True)
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error commiting_codeRepo => ", flush=True)


def run_pipeline():
    print("Starting retraining piepline...", flush=True)
    process = subprocess.Popen(
        ["./run_pipeline.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    print("Pipeline finished", flush=True)

    with model_lock:
        print("reloading model after retrain...", flush=True)
        load_model()
        commit_codeRepo("Auto update dvc.lock and metrics after retraining")


class DatasetChangeHanlder(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print("New dataset file detected", flush=True)

            threading.Thread(target=run_pipeline, daemon=True).start()


def start_watcher():
    path = "dataset"
    event_handler = DatasetChangeHanlder()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()

    while True:
        time.sleep(5)


threading.Thread(target=start_watcher, daemon=True).start()
