"""Course 185: EEG Power Spectral Density Bandpower (Theta/Beta/Alpha) Analyzer"""
import numpy as np

class EEGBandpowerAnalyzer:
    @staticmethod
    def extract_bandpower(psd: np.ndarray, freqs: np.ndarray, low_f: float, high_f: float) -> float:
        mask = (freqs >= low_f) & (freqs <= high_f)
        if np.sum(mask) <= 1:
            return 0.0
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        return float(trapz_fn(psd[mask], freqs[mask]))
