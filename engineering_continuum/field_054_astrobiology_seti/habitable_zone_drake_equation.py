"""Course 054: Circumstellar Habitable Zone & Drake Equation Biosignature Estimator"""
import numpy as np

class HabitableZoneModel:
    @staticmethod
    def calculate_hz_boundaries(stellar_luminosity_solar: float):
        inner_au = np.sqrt(stellar_luminosity_solar / 1.1)
        outer_au = np.sqrt(stellar_luminosity_solar / 0.32)
        return {"inner_hz_au": float(inner_au), "outer_hz_au": float(outer_au)}

    @staticmethod
    def drake_equation(r_star=1.5, f_p=0.9, n_e=0.4, f_l=0.5, f_i=0.2, f_c=0.2, l_years=10000.0):
        n_civilizations = r_star * f_p * n_e * f_l * f_i * f_c * l_years
        return float(n_civilizations)
