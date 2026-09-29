"""Course 017: Hong-Ou-Mandel Quantum Interference Sim"""
import numpy as np

class HOMInterferometer:
    def __init__(self, photon_bandwidth_thz=5.0):
        self.bandwidth = photon_bandwidth_thz

    def coincidence_probability(self, time_delay_ps: float) -> float:
        # HOM Dip: P_coinc = 0.5 * (1 - exp(- (time_delay * bandwidth)^2 ))
        dip = np.exp(- (time_delay_ps * self.bandwidth * 1e-1) ** 2)
        return float(0.5 * (1.0 - dip))
