"""
Engine 62: Physarum Polycephalum Protoplasmic Peristalsis + Urban Disaster Evacuation.
"""
import numpy as np

class PhysarumEvacuationRouter:
    def __init__(self, num_nodes: int = 10):
        self.n = num_nodes
        self.conductance = np.ones((self.n, self.n)) * 0.5

    def step_tubule_adaptation(self, pressure_gradients: np.ndarray, decay_rate: float = 0.05) -> np.ndarray:
        # Physarum tubule flux: Q_ij = D_ij / L_ij * (p_i - p_j)
        flux = self.conductance * pressure_gradients
        # Adaptation equation: dD/dt = |Q| - gamma * D
        d_d = np.abs(flux) - decay_rate * self.conductance
        self.conductance = np.clip(self.conductance + 0.1 * d_d, 0.01, 10.0)
        return self.conductance
