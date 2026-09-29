"""Course 416: Normalized Temperature-Scaled Cross Entropy Contrastive Learning Loss"""
import numpy as np

class SimCLRInfoNCELoss:
    def __init__(self, parameter: float = 1.0, length: int = 5):
        self.param = parameter
        self.length = length
        self.buffer = np.zeros(length)

    def execute_transform(self, data_in: np.ndarray = None) -> np.ndarray:
        arr = np.ones(self.length) if data_in is None else np.array(data_in, dtype=float)
        transformed = np.cos(arr * self.param) * self.param
        self.buffer = transformed
        return transformed

    def get_norm(self) -> float:
        return float(np.linalg.norm(self.buffer))

    def verify_properties(self) -> bool:
        return bool(np.all(np.isfinite(self.buffer)))
