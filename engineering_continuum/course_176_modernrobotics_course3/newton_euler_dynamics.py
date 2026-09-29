"""Course 176: Recursive Newton-Euler Inverse Dynamics for Serial Robotic Chains"""
import numpy as np

class NewtonEulerDynamics:
    @staticmethod
    def link_force_torque(mass: float, inertia: np.ndarray, accel_linear: np.ndarray, omega: np.ndarray, alpha: np.ndarray) -> tuple:
        force = mass * accel_linear
        torque = inertia @ alpha + np.cross(omega, inertia @ omega)
        return force, torque
