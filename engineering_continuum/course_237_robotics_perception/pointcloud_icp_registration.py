"""Course 237: Point Cloud Iterative Closest Point (ICP) SVD Rigid Registration"""
import numpy as np

class PointCloudICP:
    @staticmethod
    def align_svd(p_src: np.ndarray, p_dst: np.ndarray) -> tuple:
        # Centroids
        mu_src = np.mean(p_src, axis=0)
        mu_dst = np.mean(p_dst, axis=0)
        
        q_src = p_src - mu_src
        q_dst = p_dst - mu_dst
        
        h = q_src.T @ q_dst
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
            
        t = mu_dst - r @ mu_src
        return r, t
