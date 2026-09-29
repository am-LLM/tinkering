"""Course 156: Feature Store Schema & Statistical Data Anomaly Validator"""
import numpy as np

class FeatureStoreValidator:
    @staticmethod
    def validate_schema(data_dict: dict, expected_types: dict) -> list:
        errors = []
        for feat, exp_type in expected_types.items():
            if feat not in data_dict:
                errors.append(f"Missing feature: {feat}")
            elif not isinstance(data_dict[feat], exp_type):
                errors.append(f"Type mismatch for {feat}: expected {exp_type}")
        return errors

    @staticmethod
    def check_null_fraction(values: list, max_allowed: float = 0.05) -> bool:
        null_count = sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))
        return (null_count / len(values)) <= max_allowed if values else True
