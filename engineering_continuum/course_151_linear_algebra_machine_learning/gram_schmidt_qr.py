"""Course 151: Modified Gram-Schmidt QR Matrix Decomposition"""
import numpy as np

class GramSchmidtQR:
    @staticmethod
    def qr_decompose(a: np.ndarray) -> tuple:
        m, n = a.shape
        q = np.zeros((m, n), dtype=np.float64)
        r = np.zeros((n, n), dtype=np.float64)
        v = a.astype(np.float64).copy()
        
        for i in range(n):
            r[i, i] = np.linalg.norm(v[:, i])
            q[:, i] = v[:, i] / r[i, i]
            for j in range(i + 1, n):
                r[i, j] = np.dot(q[:, i], v[:, j])
                v[:, j] = v[:, j] - r[i, j] * q[:, i]
                
        return q, r
