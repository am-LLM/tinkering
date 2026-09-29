"""Course 063: Lorenz Strange Attractor & Sensitive Dependence on Initial Conditions"""
import numpy as np

class LorenzChaosSimulator:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma, self.rho, self.beta = sigma, rho, beta

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])

    def step_rk4(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def compute_trajectory_divergence(self, steps=3000, dt=0.01, eps=1e-3) -> float:
        # Start near attractor
        s1 = np.array([10.0, 10.0, 25.0])
        s2 = s1 + np.array([eps, 0.0, 0.0])
        for _ in range(steps):
            s1 = self.step_rk4(s1, dt)
            s2 = self.step_rk4(s2, dt)
        dist = np.linalg.norm(s1 - s2)
        return float(dist)
