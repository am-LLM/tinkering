"""
Engine 64: Parity-Time Symmetric Non-Hermitian Exceptional Point + Microfluidic Sepsis Monitor.
"""
import numpy as np

class ExceptionalPointSepsisDetector:
    def __init__(self, coupling_kappa: float = 1.0):
        self.kappa = coupling_kappa

    def compute_ep_eigenvalue_splitting(self, endotoxin_perturbation_eps: float) -> float:
        # Non-Hermitian 2x2 Hamiltonian: H = [[omega_0 - i*gamma, kappa], [kappa, omega_0 + i*gamma]]
        # Eigenvalue splitting near EP2: delta_lambda = 2 * sqrt(kappa * eps)
        splitting = 2.0 * np.sqrt(max(0.0, self.kappa * endotoxin_perturbation_eps))
        sensitivity_enhancement = float(splitting / max(1e-9, endotoxin_perturbation_eps))
        return float(sensitivity_enhancement)
