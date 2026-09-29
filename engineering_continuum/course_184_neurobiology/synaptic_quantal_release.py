"""Course 184: Synaptic Binomial Quantal Neurotransmitter Release Model"""
import math

class SynapticQuantalRelease:
    @staticmethod
    def binomial_release_probability(n_vesicles: int, p_release: float, k_success: int) -> float:
        comb = math.comb(n_vesicles, k_success)
        return float(comb * (p_release ** k_success) * ((1.0 - p_release) ** (n_vesicles - k_success)))

    @staticmethod
    def mean_quantal_content(n_vesicles: int, p_release: float) -> float:
        return float(n_vesicles * p_release)
