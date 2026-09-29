"""Course 215: Linear Support Vector Machine Dual Sequential Minimal Optimization"""
import numpy as np

class LinearSVMSMO:
    def __init__(self, c: float = 1.0):
        self.c = c
        self.w = None
        self.b = 0.0

    def fit_linear(self, x: np.ndarray, y: np.ndarray, max_passes: int = 5):
        # Simplified linear dual weights
        m, n = x.shape
        self.w = np.zeros(n)
        alphas = np.zeros(m)
        for _ in range(max_passes):
            for i in range(m):
                margin = y[i] * (np.dot(self.w, x[i]) + self.b)
                if margin < 1.0:
                    self.w += 0.01 * (y[i] * x[i] - 0.01 * self.w)
                    self.b += 0.01 * y[i]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x @ self.w + self.b)
