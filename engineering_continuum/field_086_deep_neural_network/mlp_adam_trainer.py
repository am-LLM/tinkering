"""Course 086: Multi-Layer Perceptron (MLP) Forward & Adam Optimizer Step"""
import numpy as np

class MLPAdamTrainer:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 0.01):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2

    def compute_mse_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true)**2))
