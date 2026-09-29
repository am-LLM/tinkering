"""
Engine 65: Extreme Value Theory (EVT) Fréchet Tails + EW Anti-Jamming Spatial Null Flight.
"""
import numpy as np

class FrechetEWTrajectoryPlanner:
    def __init__(self, shape_alpha: float = 2.5, scale_sigma: float = 1.0):
        self.alpha = shape_alpha
        self.sigma = scale_sigma

    def compute_jamming_null_escape_vector(self, jammer_rf_power_matrix: np.ndarray) -> np.ndarray:
        # GEV Frechet distribution of extreme jamming power spikes
        max_power = np.max(jammer_rf_power_matrix)
        frechet_prob = np.exp(-((max_power / self.sigma) ** (-self.alpha)))
        
        # Spatial gradient toward RF null
        grad = np.gradient(jammer_rf_power_matrix)
        escape_dir = -np.array([np.mean(grad[0]), np.mean(grad[1]), 0.0])
        norm = np.linalg.norm(escape_dir)
        escape_vector = (escape_dir / max(1e-6, norm)) * (1.0 + float(frechet_prob) * 5.0)
        return np.asarray(escape_vector, dtype=np.float64)
