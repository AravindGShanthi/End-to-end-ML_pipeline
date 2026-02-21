from feast import FeatureStore

store = FeatureStore(repo_path="features")

feature_vector = store.get_online_features(
    features=["student_features:G1"],
    entity_rows=[{"student_id": 1}],
).to_dict()

print(feature_vector)
