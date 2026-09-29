"""Course 098: Lorenz Chaotic Attractor Integrator and Trajectory Analyzer"""
import numpy as np

class LorenzChaosAnalyzer:
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        self.sigma, self.rho, self.beta = sigma, rho, beta

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])

    def rk4_step(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
