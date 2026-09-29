"""Course 222: Hong-Ou-Mandel (HOM) Two-Photon Quantum Interference Dip Calculator"""
import math

class HongOuMandelDip:
    @staticmethod
    def hom_visibility(coincidence_min: float, coincidence_max: float) -> float:
        # Visibility V = (C_max - C_min) / C_max
        if coincidence_max <= 0:
            return 0.0
        return float((coincidence_max - coincidence_min) / coincidence_max)

    @staticmethod
    def gaussian_dip_profile(delay_fs: float, coherence_time_fs: float = 100.0, visibility: float = 0.95) -> float:
        # P_coincidence(tau) = 0.5 * (1 - V * exp(-(tau / tau_c)^2))
        return float(0.5 * (1.0 - visibility * math.exp(-(delay_fs / coherence_time_fs) ** 2)))
