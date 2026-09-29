"""Course 174: Modern Robotics SE(3) Homogeneous Transformation Matrix Engine"""
import numpy as np

class SE3Transform:
    @staticmethod
    def build_t(r_mat: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
        t = np.eye(4)
        t[0:3, 0:3] = r_mat
        t[0:3, 3] = p_vec
        return t

    @staticmethod
    def inverse_t(t_mat: np.ndarray) -> np.ndarray:
        r = t_mat[0:3, 0:3]
        p = t_mat[0:3, 3]
        t_inv = np.eye(4)
        t_inv[0:3, 0:3] = r.T
        t_inv[0:3, 3] = -r.T @ p
        return t_inv
