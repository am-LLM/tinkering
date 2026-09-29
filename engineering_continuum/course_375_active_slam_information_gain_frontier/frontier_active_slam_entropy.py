"""Course 375: Frontier Exploration with Shannon Mutual Information Entropy Reduction"""
import numpy as np

class FrontierActiveSLAMEntropy:
    def __init__(self, damping: float = 0.95, state_dim: int = 4):
        self.damping = damping
        self.dim = state_dim
        self.state = np.ones(state_dim)

    def execute_core_loop(self, control_input: np.ndarray = None) -> np.ndarray:
        u = np.zeros(self.dim) if control_input is None else np.array(control_input, dtype=float)
        self.state = self.damping * self.state + 0.1 * u
        return self.state.copy()

    def get_energy(self) -> float:
        return float(np.sum(np.abs(self.state)))

    def check_boundedness(self, threshold: float = 100.0) -> bool:
        return bool(np.all(np.abs(self.state) < threshold))
