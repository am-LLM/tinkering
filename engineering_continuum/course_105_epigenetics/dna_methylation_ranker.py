"""Course 105: DNA Methylation Beta-to-M Value Converter & DMR Ranker"""
import numpy as np

class DNAMethylationRanker:
    @staticmethod
    def beta_to_m(beta: np.ndarray, offset: float = 1e-4) -> np.ndarray:
        b_clamped = np.clip(beta, offset, 1.0 - offset)
        return np.log2(b_clamped / (1.0 - b_clamped))

    @staticmethod
    def differential_rank(case_m: np.ndarray, control_m: np.ndarray) -> np.ndarray:
        delta = np.mean(case_m, axis=1) - np.mean(control_m, axis=1)
        return np.argsort(-np.abs(delta))
