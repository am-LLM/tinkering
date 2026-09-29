"""Course 376: AC Power Flow Solver using Newton-Raphson & Bus Admittance Matrix"""
import numpy as np

class PowerFlowNewtonRaphson:
    def __init__(self, y_bus: np.ndarray):
        self.y_bus = np.array(y_bus, dtype=complex)
        self.n_bus = len(self.y_bus)
        self.g = np.real(self.y_bus)
        self.b = np.imag(self.y_bus)

    def calculate_power_injections(self, v_mag: np.ndarray, v_ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.zeros(self.n_bus)
        q = np.zeros(self.n_bus)
        for i in range(self.n_bus):
            for j in range(self.n_bus):
                theta_ij = v_ang[i] - v_ang[j]
                p[i] += v_mag[i] * v_mag[j] * (self.g[i, j] * np.cos(theta_ij) + self.b[i, j] * np.sin(theta_ij))
                q[i] += v_mag[i] * v_mag[j] * (self.g[i, j] * np.sin(theta_ij) - self.b[i, j] * np.cos(theta_ij))
        return p, q

    def power_mismatch(self, v_mag: np.ndarray, v_ang: np.ndarray, p_spec: np.ndarray, q_spec: np.ndarray) -> np.ndarray:
        p_calc, q_calc = self.calculate_power_injections(v_mag, v_ang)
        dp = p_spec - p_calc
        dq = q_spec - q_calc
        return np.concatenate([dp, dq])
