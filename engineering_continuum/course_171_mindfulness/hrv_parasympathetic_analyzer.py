"""Course 171: Heart Rate Variability (HRV) RMSSD & Parasympathetic Recovery Analyzer"""
import numpy as np

class HRVParasympatheticAnalyzer:
    @staticmethod
    def calculate_rmssd(rr_intervals_ms: np.ndarray) -> float:
        diffs = np.diff(rr_intervals_ms)
        return float(np.sqrt(np.mean(diffs ** 2)))

    @staticmethod
    def calculate_pnn50(rr_intervals_ms: np.ndarray) -> float:
        diffs = np.abs(np.diff(rr_intervals_ms))
        nn50 = np.sum(diffs > 50.0)
        return float(nn50 / len(diffs)) if len(diffs) > 0 else 0.0
