import numpy as np

class DrakeExtremophileModel:
    def __init__(self, r_star: float, f_planets: float, n_habitable: float):
        self.base_rate = r_star * f_planets * n_habitable

    def estimate_active_biosignatures(self, f_life: float, f_extremophile_resilience: float, bio_lifespan: float) -> float:
        n_signatures = self.base_rate * f_life * f_extremophile_resilience * bio_lifespan
        return float(max(0.0, n_signatures))
