"""Course 226: Logistic Regression Logit Link & Odds Ratio Gradient Engine"""
import numpy as np

class LogisticRegressionLogit:
    def __init__(self, lr: float = 0.1, n_iter: int = 100):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.n_iter):
            linear = np.dot(x, self.weights) + self.bias
            y_pred = 1.0 / (1.0 + np.exp(-np.clip(linear, -20, 20)))
            
            dw = (1.0 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1.0 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def odds_ratio(self, feature_idx: int) -> float:
        return float(np.exp(self.weights[feature_idx]))
