"""Course 040: 8-Point RANSAC Essential Matrix & Visual Odometry Pose Recovery"""
import numpy as np

class VisualOdometryRANSAC:
    def __init__(self, k_matrix=None, max_iterations=100, threshold=1e-3):
        self.k = k_matrix if k_matrix is not None else np.eye(3)
        self.max_iter = max_iterations
        self.threshold = threshold

    def compute_essential_matrix(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        # Normalized coordinates: x' = K^-1 * x
        k_inv = np.linalg.inv(self.k)
        n = p1.shape[0]
        a = np.zeros((n, 9))
        for i in range(n):
            x1 = k_inv @ np.array([p1[i, 0], p1[i, 1], 1.0])
            x2 = k_inv @ np.array([p2[i, 0], p2[i, 1], 1.0])
            a[i] = [
                x2[0]*x1[0], x2[0]*x1[1], x2[0],
                x2[1]*x1[0], x2[1]*x1[1], x2[1],
                x1[0],       x1[1],       1.0
            ]
        _, _, vh = np.linalg.svd(a)
        e = vh[-1].reshape(3, 3)
        # Enforce rank-2 constraint: singular values (s, s, 0)
        u, s, vt = np.linalg.svd(e)
        s_enforced = np.diag([(s[0] + s[1]) / 2.0, (s[0] + s[1]) / 2.0, 0.0])
        return u @ s_enforced @ vt

    def estimate_pose(self, p1: np.ndarray, p2: np.ndarray):
        e = self.compute_essential_matrix(p1, p2)
        u, _, vt = np.linalg.svd(e)
        w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        r = u @ w.T @ vt
        if np.linalg.det(r) < 0:
            r = -r
        t = u[:, 2]
        return r, t
