from feature_store_validator import FeatureStoreValidator

def test_feature_validator():
    errs = FeatureStoreValidator.validate_schema({"age": 25, "income": "high"}, {"age": int, "income": float})
    assert len(errs) == 1
    assert FeatureStoreValidator.check_null_fraction([1.0, None, 2.0], 0.5) is True
