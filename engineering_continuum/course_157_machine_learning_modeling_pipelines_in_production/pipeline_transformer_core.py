"""Course 157: Production Machine Learning Pipeline Transformer Core"""
import numpy as np

class StandardScaleTransformer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
