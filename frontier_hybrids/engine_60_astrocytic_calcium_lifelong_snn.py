"""
Engine 60: Astrocytic Glial Calcium Wave Dynamics + Lifelong Spiking Graph SNN.
"""
import numpy as np

class AstrocyticLifelongSNN:
    def __init__(self, num_nodes: int = 16):
        self.n = num_nodes
        self.ca_level = np.zeros(self.n)
        self.synaptic_weights = np.random.uniform(0.4, 0.8, (self.n, self.n))

    def propagate_glial_calcium_wave(self, neural_spikes: np.ndarray, diffusion_coeff: float = 0.15) -> np.ndarray:
        # IP3-mediated calcium wave diffusion: dCa/dt = D * laplacian(Ca) + k_prod * spikes
        laplacian = np.roll(self.ca_level, 1) + np.roll(self.ca_level, -1) - 2.0 * self.ca_level
        d_ca = diffusion_coeff * laplacian + 0.3 * neural_spikes - 0.05 * self.ca_level
        self.ca_level = np.clip(self.ca_level + d_ca, 0.0, 1.0)
        
        # Gliotransmission modulates synaptic plasticity bounds, preventing catastrophic forgetting
        plasticity_gate = 1.0 / (1.0 + np.exp(-10.0 * (self.ca_level - 0.5)))
        return plasticity_gate
