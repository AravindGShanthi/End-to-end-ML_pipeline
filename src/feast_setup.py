import os
from datetime import datetime, timedelta, timezone

from feast import (
    Entity,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float64, Int64

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "features")
FEATURE_DATA_PATH = os.getenv(
    "FEATURE_DATA_PATH", "../data/processed/student_features.parquet"
)
MATERIALIZE_ON_APPLY = os.getenv("MATERIALIZE_ON_APPLY", "true").lower() == "true"


def build_definitions():
    student = Entity(name="student_id", join_keys=["student_id"])

    student_source = FileSource(
        path=FEATURE_DATA_PATH,
        timestamp_field="event_timestamp",
    )

    student_features = FeatureView(
        name="student_features",
        entities=[student],
        ttl=timedelta(days=365),
        schema=[
            Field(name="fail_abs", dtype=Int64),
            Field(name="G1", dtype=Int64),
            Field(name="failures", dtype=Int64),
            Field(name="abs_log", dtype=Float64),
            Field(name="studytime", dtype=Int64),
            Field(name="Medu", dtype=Int64),
            Field(name="Fedu", dtype=Int64),
            Field(name="age", dtype=Int64),
        ],
        source=student_source,
    )

    return [student, student_features]


def apply_definitions(store: FeatureStore):
    objects = build_definitions()
    store.apply(objects)


# def materialize_incremental(store: FeatureStore):
#     end_time = datetime.now(timezone.utc)

#     store.materialize_incremental(end_date=end_time)
# store.materialize(
#     start_date=datetime(2000, 1, 1, tzinfo=timezone.utc),
#     end_date=end_time,
# )


def main():
    try:
        store = FeatureStore(repo_path=FEAST_REPO_PATH)

        apply_definitions(store)

        # if MATERIALIZE_ON_APPLY:
        #     materialize_incremental(store)

    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    main()
