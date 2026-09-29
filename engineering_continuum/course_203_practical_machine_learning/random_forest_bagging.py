"""Course 203: Random Forest Decision Stump Bootstrap Ensemble Classifier"""
import numpy as np

class BaggedStumpClassifier:
    def __init__(self, n_estimators: int = 10):
        self.n = n_estimators
        self.stumps = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples = x.shape[0]
        for _ in range(self.n):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            feat = np.random.choice(x.shape[1])
            thresh = np.median(x[idx, feat])
            pred_left = 1 if np.mean(y[idx][x[idx, feat] <= thresh]) > 0.5 else 0
            self.stumps.append((feat, thresh, pred_left))

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = []
        for feat, thresh, pred_left in self.stumps:
            p = np.where(x[:, feat] <= thresh, pred_left, 1 - pred_left)
            preds.append(p)
        return (np.mean(preds, axis=0) >= 0.5).astype(int)
