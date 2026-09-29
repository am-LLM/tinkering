"""Course 179: Quintic Polynomial Trajectory Generator for Smooth Autonomous Navigation"""
import numpy as np

class QuinticPolynomialPlanner:
    @staticmethod
    def solve_boundary(x0: float, v0: float, a0: float, x1: float, v1: float, a1: float, t: float) -> np.ndarray:
        # Solve for coefficients [c0, c1, c2, c3, c4, c5]
        m = np.array([
            [t**3,   t**4,    t**5],
            [3*t**2, 4*t**3,  5*t**4],
            [6*t,    12*t**2, 20*t**3]
        ])
        b = np.array([
            x1 - (x0 + v0*t + 0.5*a0*t**2),
            v1 - (v0 + a0*t),
            a1 - a0
        ])
        c345 = np.linalg.solve(m, b)
        return np.array([x0, v0, 0.5*a0, c345[0], c345[1], c345[2]])
