"""Course 188: Multi-Variable Newton-Raphson Nonlinear System Solver"""
import numpy as np

class MultivariateNewtonRaphson:
    @staticmethod
    def solve(f_system, jacobian_func, x0: np.ndarray, max_iter: int = 20, tol: float = 1e-6) -> np.ndarray:
        x = x0.astype(np.float64).copy()
        for _ in range(max_iter):
            f_val = f_system(x)
            if np.linalg.norm(f_val) < tol:
                break
            j_mat = jacobian_func(x)
            delta = np.linalg.solve(j_mat, -f_val)
            x += delta
            if np.linalg.norm(delta) < tol:
                break
        return x
