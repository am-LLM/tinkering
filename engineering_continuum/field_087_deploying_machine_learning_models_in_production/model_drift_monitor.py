"""Course 087: Real-Time Kolmogorov-Smirnov Distribution Drift Detector"""
import numpy as np

class DriftMonitor:
    @staticmethod
    def ks_statistic(sample1: np.ndarray, sample2: np.ndarray) -> float:
        s1 = np.sort(sample1)
        s2 = np.sort(sample2)
        all_vals = np.sort(np.concatenate([s1, s2]))
        cdf1 = np.searchsorted(s1, all_vals, side='right') / len(s1)
        cdf2 = np.searchsorted(s2, all_vals, side='right') / len(s2)
        return float(np.max(np.abs(cdf1 - cdf2)))

    def is_drift_detected(self, baseline: np.ndarray, current: np.ndarray, threshold: float = 0.3) -> bool:
        ks = self.ks_statistic(baseline, current)
        return ks > threshold
