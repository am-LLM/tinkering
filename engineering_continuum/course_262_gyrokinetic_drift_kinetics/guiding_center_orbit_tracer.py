"""Course 262: Guiding Center Gyrokinetic Particle Orbit Integration"""
import numpy as np

class GuidingCenterOrbitTracer:
    def __init__(self, param: float = 1.0, dimension: int = 4):
        self.param = param
        self.dim = dimension
        self.state = np.zeros(dimension)

    def compute_metric(self, input_vector: np.ndarray = None) -> float:
        v = np.ones(self.dim) if input_vector is None else np.array(input_vector, dtype=float)
        val = np.sum(v**2) * self.param
        return float(val)

    def step_dynamics(self, dt: float = 0.01) -> np.ndarray:
        self.state += dt * (self.param - self.state)
        return self.state.copy()

    def evaluate_invariant(self) -> bool:
        return np.all(np.isfinite(self.state))
