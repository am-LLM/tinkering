"""Course 045: Lindblad Master Equation & Two-Level Open Quantum System Dissipation"""
import numpy as np

class LindbladOpenQuantumSystem:
    def __init__(self, omega_0=1.0, gamma_dephasing=0.05, gamma_relaxation=0.02):
        self.omega = omega_0
        self.gamma_phi = gamma_dephasing
        self.gamma_1 = gamma_relaxation
        
        # Pauli matrices
        self.sx = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sz = np.array([[1, 0], [0, -1]], dtype=complex)
        self.sp = np.array([[0, 1], [0, 0]], dtype=complex)
        self.sm = np.array([[0, 0], [1, 0]], dtype=complex)
        
        # Hamiltonian H = 0.5 * omega * sigma_z
        self.h = 0.5 * self.omega * self.sz
        # Initial pure superposition state |+> = (|0> + |1>) / sqrt(2)
        psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)
        self.rho = np.outer(psi0, np.conj(psi0))

    def step(self, dt: float = 0.01):
        # Von Neumann commutator -i[H, rho]
        coherent = -1j * (self.h @ self.rho - self.rho @ self.h)
        
        # Lindblad dephasing dissipator: L_phi = sqrt(gamma_phi/2) * sigma_z
        lindblad_phi = (self.sz @ self.rho @ self.sz - self.rho) * (self.gamma_phi / 2.0)
        
        # Lindblad relaxation dissipator: L_1 = sqrt(gamma_1) * sigma_minus
        lindblad_relax = self.gamma_1 * (self.sm @ self.rho @ self.sp - 0.5 * (self.sp @ self.sm @ self.rho + self.rho @ self.sp @ self.sm))
        
        d_rho = coherent + lindblad_phi + lindblad_relax
        self.rho += d_rho * dt
        # Enforce trace=1 and Hermiticity
        self.rho = 0.5 * (self.rho + np.conj(self.rho).T)
        self.rho /= np.trace(self.rho)
        return float(np.real(np.trace(self.rho @ self.rho))) # Purity Tr(rho^2)
