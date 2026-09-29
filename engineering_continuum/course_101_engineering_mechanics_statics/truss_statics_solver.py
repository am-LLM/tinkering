"""Course 101: 2D Planar Truss Static Equilibrium Joint Force Solver"""
import numpy as np

class TrussStaticsSolver:
    @staticmethod
    def solve_joint_2d(fx_external: float, fy_external: float, angles_rad: list) -> np.ndarray:
        # Sum Fx = 0, Sum Fy = 0 for 2 concurrent member forces
        theta1, theta2 = angles_rad[0], angles_rad[1]
        a = np.array([
            [np.cos(theta1), np.cos(theta2)],
            [np.sin(theta1), np.sin(theta2)]
        ])
        b = np.array([-fx_external, -fy_external])
        return np.linalg.solve(a, b)
