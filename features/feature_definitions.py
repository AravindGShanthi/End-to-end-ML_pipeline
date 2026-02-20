from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float64, Int64

student = Entity(name="student_id", join_keys=["student_id"])

student_source = FileSource(
    path="../data/processed/student_features.parquet",
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
