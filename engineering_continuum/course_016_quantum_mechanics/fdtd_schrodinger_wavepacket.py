"""Course 016: 1D Time-Dependent Schrodinger Wave-Packet Solver"""
import numpy as np

class SchrodingerFDTD:
    def __init__(self, grid_size=200, dx=0.1, dt=0.005):
        self.n, self.dx, self.dt = grid_size, dx, dt
        self.x = np.linspace(-10, 10, grid_size)
        self.psi = np.exp(-(self.x)**2) * np.exp(1j * 3.0 * self.x)
        self.psi /= np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        self.v = np.zeros(grid_size)

    def step(self):
        d2_psi = (np.roll(self.psi, -1) - 2 * self.psi + np.roll(self.psi, 1)) / (self.dx**2)
        h_psi = -0.5 * d2_psi + self.v * self.psi
        self.psi -= 1j * h_psi * self.dt
        norm = np.sum(np.abs(self.psi)**2) * self.dx
        self.psi /= np.sqrt(norm)
        return float(np.real(np.sum(self.x * np.abs(self.psi)**2) * self.dx))
