"""Course 219: 1D Time-Dependent Schrödinger Equation FDTD Wave Packet Integrator"""
import numpy as np

class Schrodinger1DFDTD:
    def __init__(self, num_points: int = 100, dx: float = 0.1, dt: float = 0.001, mass: float = 1.0, hbar: float = 1.0):
        self.n = num_points
        self.dx, self.dt = dx, dt
        self.m, self.hbar = mass, hbar
        self.v = np.zeros(num_points)

    def step(self, psi: np.ndarray) -> np.ndarray:
        # Kinetic operator second derivative
        d2_psi = np.zeros_like(psi)
        d2_psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / (self.dx ** 2)
        h_psi = -(self.hbar ** 2 / (2.0 * self.m)) * d2_psi + self.v * psi
        # i * hbar * dpsi/dt = H * psi  => dpsi = -i / hbar * H * psi * dt
        psi_next = psi - 1j * (h_psi / self.hbar) * self.dt
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi_next)**2) * self.dx)
        return psi_next / norm if norm > 0 else psi_next
