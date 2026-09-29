"""Course 083: Z-Score & Interquartile Range (IQR) Statistical Anomaly Detector"""
import numpy as np

class StatisticalAnomalyDetector:
    @staticmethod
    def z_score_anomalies(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros(len(data), dtype=bool)
        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold

    @staticmethod
    def iqr_anomalies(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (factor * iqr)
        upper_bound = q75 + (factor * iqr)
        return (data < lower_bound) | (data > upper_bound)
