"""Course 180: Newton-Raphson Optimization Step with Hessian Matrix Inverse"""
import numpy as np

class HessianOptimizer:
    @staticmethod
    def newton_step(x: np.ndarray, grad: np.ndarray, hessian: np.ndarray) -> np.ndarray:
        inv_hess = np.linalg.inv(hessian)
        return x - inv_hess @ grad
