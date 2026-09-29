"""Course 149: Population Stability Index (PSI) Production Data Drift Detector"""
import numpy as np

class PSIDriftDetector:
    @staticmethod
    def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
        # Quantile binning on baseline
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(baseline, percentiles)
        bins[0] -= 1e-5
        bins[-1] += 1e-5
        
        base_counts, _ = np.histogram(baseline, bins=bins)
        targ_counts, _ = np.histogram(target, bins=bins)
        
        base_pct = np.maximum(base_counts / len(baseline), 1e-4)
        targ_pct = np.maximum(targ_counts / len(target), 1e-4)
        
        psi = np.sum((targ_pct - base_pct) * np.log(targ_pct / base_pct))
        return float(psi)
