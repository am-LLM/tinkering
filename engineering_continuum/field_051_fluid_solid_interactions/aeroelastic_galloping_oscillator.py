"""Course 051: 2-DoF Aeroelastic Wing Flutter & Galloping Non-Linear Solver"""
import numpy as np

class AeroelasticFlutterModel:
    def __init__(self, m=1.5, i_alpha=0.05, k_h=150.0, k_alpha=40.0, b=0.25):
        self.m, self.i_alpha, self.k_h, self.k_alpha, self.b = m, i_alpha, k_h, k_alpha, b
        self.state = np.array([0.0, 0.0, 0.05, 0.0])

    def derivatives(self, state: np.ndarray, airspeed_u: float, rho: float = 1.225):
        h, h_dot, alpha, alpha_dot = state
        lift = 0.5 * rho * (airspeed_u ** 2) * (2 * self.b) * 2.0 * np.pi * (alpha + h_dot / max(airspeed_u, 1e-3))
        moment = 0.5 * rho * (airspeed_u ** 2) * ((2 * self.b) ** 2) * 0.5 * alpha
        
        dh = h_dot
        dh_dot = (-self.k_h * h - lift) / self.m
        dalpha = alpha_dot
        dalpha_dot = (-self.k_alpha * alpha + moment) / self.i_alpha
        return np.array([dh, dh_dot, dalpha, dalpha_dot])

    def step(self, airspeed_u: float, dt: float = 0.001):
        k1 = self.derivatives(self.state, airspeed_u)
        k2 = self.derivatives(self.state + 0.5 * dt * k1, airspeed_u)
        k3 = self.derivatives(self.state + 0.5 * dt * k2, airspeed_u)
        k4 = self.derivatives(self.state + dt * k3, airspeed_u)
        self.state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return float(self.state[2])
