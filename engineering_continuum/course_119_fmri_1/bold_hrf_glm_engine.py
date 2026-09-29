"""Course 119: BOLD Hemodynamic Response Function (HRF) Double-Gamma Convolver"""
import numpy as np

class BOLDHRFGLMEngine:
    @staticmethod
    def spm_canonical_hrf(tr: float = 2.0, length_sec: float = 32.0) -> np.ndarray:
        t = np.arange(0, length_sec, tr)
        # Double gamma parameters
        a1, b1 = 6.0, 1.0
        a2, b2 = 16.0, 1.0
        c = 1.0 / 6.0
        d1 = (t ** (a1 - 1)) * np.exp(-t / b1)
        d2 = (t ** (a2 - 1)) * np.exp(-t / b2)
        hrf = d1 - c * d2
        return hrf / np.sum(hrf)

    @classmethod
    def convolve_stimulus(cls, stimulus_train: np.ndarray, tr: float = 2.0) -> np.ndarray:
        hrf = cls.spm_canonical_hrf(tr)
        return np.convolve(stimulus_train, hrf, mode='same')
