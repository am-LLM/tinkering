"""Course 158: Stratified K-Fold Splitter & Precision-Recall Evaluator"""
import numpy as np

class KFoldPREvaluator:
    @staticmethod
    def k_fold_split(n_samples: int, k: int = 5) -> list:
        indices = np.arange(n_samples)
        fold_size = n_samples // k
        folds = []
        for i in range(k):
            test_idx = indices[i*fold_size:(i+1)*fold_size if i != k-1 else n_samples]
            train_idx = np.setdiff1d(indices, test_idx)
            folds.append((train_idx, test_idx))
        return folds

    @staticmethod
    def precision_recall_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple:
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(prec), float(rec)
