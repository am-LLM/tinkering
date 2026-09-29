"""Course 065: PN Junction Built-in Potential & Shockley Diode IV Engine"""
import numpy as np

class PNJunctionModel:
    def __init__(self, na_cm3=1e16, nd_cm3=1e17, ni_cm3=1.5e10, t_kelvin=300.0):
        k_b = 1.380649e-23
        q = 1.60217663e-19
        self.v_t = (k_b * t_kelvin) / q
        self.v_bi = self.v_t * np.log((na_cm3 * nd_cm3) / (ni_cm3 ** 2))
        self.i_s = 1e-14

    def shockley_current(self, v_applied: float) -> float:
        v_clipped = min(v_applied, 0.9)
        return float(self.i_s * (np.exp(v_clipped / self.v_t) - 1.0))
