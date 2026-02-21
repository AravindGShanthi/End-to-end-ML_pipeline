import os
from datetime import datetime, timezone

from feast import FeatureStore

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "features")
MARKER_FILE_PATH = "data/feast_materialized.marker"


def materialize_incremental(store: FeatureStore):
    end_time = datetime.now(timezone.utc)

    store.materialize_incremental(end_date=end_time)

    return end_time


def write_marker(timestamp: datetime):
    os.makedirs(os.path.dirname(MARKER_FILE_PATH), exist_ok=True)

    with open(MARKER_FILE_PATH, "w") as f:
        f.write(timestamp.isoformat())


def main():
    try:
        print("Feast repo path: ", FEAST_REPO_PATH)
        store = FeatureStore(repo_path=FEAST_REPO_PATH)

        end_time = materialize_incremental(store)
        print("Feast end_time: ", end_time)

        write_marker(end_time)
    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    main()
