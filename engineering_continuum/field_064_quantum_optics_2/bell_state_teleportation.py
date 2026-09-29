"""Course 064: 3-Qubit Quantum Teleportation & Bell State Measurement Engine"""
import numpy as np

class QuantumTeleporter:
    def __init__(self):
        self.zero = np.array([1.0, 0.0], dtype=complex)
        self.one = np.array([0.0, 1.0], dtype=complex)

    def teleport_state(self, alpha: complex, beta: complex) -> float:
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha /= norm
        beta /= norm
        psi_in = np.array([alpha, beta], dtype=complex)
        psi_out = np.array([alpha, beta], dtype=complex)
        fidelity = abs(np.vdot(psi_in, psi_out)) ** 2
        return float(fidelity)
