"""Course 230: LLC Resonant Converter First Harmonic Approximation (FHA) Voltage Gain"""
import math

class LLCResonantFHAGain:
    @staticmethod
    def voltage_gain(fn_norm_freq: float, ln_inductance_ratio: float = 5.0, q_quality_factor: float = 0.5) -> float:
        # M(fn) = 1 / sqrt( [1 + 1/Ln * (1 - 1/fn^2)]^2 + [Q * (fn - 1/fn)]^2 )
        term1 = (1.0 + (1.0 / ln_inductance_ratio) * (1.0 - 1.0 / (fn_norm_freq ** 2))) ** 2
        term2 = (q_quality_factor * (fn_norm_freq - 1.0 / fn_norm_freq)) ** 2
        return float(1.0 / math.sqrt(term1 + term2))
