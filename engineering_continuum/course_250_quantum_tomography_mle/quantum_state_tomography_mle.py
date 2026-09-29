"""Course 250: Single-Qubit Quantum State Tomography via Maximum Likelihood"""
import numpy as np

class QuantumStateTomographyMLE:
    PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def __init__(self):
        pass

    def reconstruct_density_matrix(self, exp_x: float, exp_y: float, exp_z: float) -> np.ndarray:
        # Stokes parameter reconstruction: rho = 0.5 * (I + rx*X + ry*Y + rz*Z)
        r_vec = np.array([exp_x, exp_y, exp_z])
        norm = np.linalg.norm(r_vec)
        if norm > 1.0: # Physical purity constraint projection
            r_vec = r_vec / norm
        rho = 0.5 * (self.PAULI_I + r_vec[0] * self.PAULI_X + r_vec[1] * self.PAULI_Y + r_vec[2] * self.PAULI_Z)
        return rho

    def fidelity(self, rho_true: np.ndarray, rho_est: np.ndarray) -> float:
        # For pure target state: F = Tr(rho_true @ rho_est)
        val = np.real(np.trace(rho_true @ rho_est))
        return float(np.clip(val, 0.0, 1.0))
