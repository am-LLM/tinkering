"""Course 196: Relativistic Particle Collider Mandelstam Variables (s, t, u)"""
import numpy as np

class MandelstamKinematics:
    @staticmethod
    def mandelstam_s(p1_4vec: np.ndarray, p2_4vec: np.ndarray) -> float:
        # Minkowski metric (+ - - -)
        p_tot = p1_4vec + p2_4vec
        s = p_tot[0]**2 - np.sum(p_tot[1:]**2)
        return float(s)

    @staticmethod
    def center_of_mass_energy(s: float) -> float:
        return float(np.sqrt(max(0.0, s)))
