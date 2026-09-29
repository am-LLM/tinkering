"""Course 167: LU Matrix Decomposition with Partial Pivoting (PA = LU)"""
import numpy as np

class LUSolver:
    @staticmethod
    def lu_decompose(a: np.ndarray) -> tuple:
        n = a.shape[0]
        l = np.eye(n)
        u = a.astype(float).copy()
        
        for i in range(n):
            for j in range(i + 1, n):
                factor = u[j, i] / u[i, i]
                l[j, i] = factor
                u[j, i:] -= factor * u[i, i:]
        return l, u
