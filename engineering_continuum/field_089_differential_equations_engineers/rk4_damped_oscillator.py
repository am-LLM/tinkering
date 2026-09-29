"""Course 089: 4th-Order Runge-Kutta (RK4) Damped Harmonic Oscillator ODE Solver"""
import numpy as np

class DampedOscillatorRK4:
    def __init__(self, m: float = 1.0, c: float = 0.2, k: float = 4.0):
        self.m, self.c, self.k = m, c, k

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, v = state
        dx = v
        dv = -(self.c / self.m) * v - (self.k / self.m) * x
        return np.array([dx, dv])

    def step(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = self.derivatives(state, t)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.derivatives(state + dt * k3, t + dt)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
