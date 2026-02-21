import os
import subprocess
import threading
import time
from datetime import datetime

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from feast import FeatureStore
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

FEAST_REPO_PATH = "features"
MODEL_URI = "models:/student_dropout_model@champion"

model = mlflow.pyfunc.load_model(MODEL_URI)
store = FeatureStore(repo_path=FEAST_REPO_PATH)
app = FastAPI(title="student dropout prediction API")


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

    print(feature_vector)

    prediction = model.predict(feature_vector)

    return {
        "student_id": request.student_id,
        "prediction": float(prediction[0]),
    }


class DatasetChangeHanlder(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print("New dataset file detected")
            subprocess.run(["./run_pipeline.sh"])


def start_watcher():
    path = "dataset"
    event_handler = DatasetChangeHanlder()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()

    while True:
        time.sleep(5)


threading.Thread(target=start_watcher, daemon=True).start()
