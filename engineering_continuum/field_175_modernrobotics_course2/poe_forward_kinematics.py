"""Course 175: Product of Exponentials (PoE) Forward Kinematics for Serial Manipulators"""
import numpy as np

class PoEForwardKinematics:
    @staticmethod
    def matrix_exp6(screw_axis: np.ndarray, theta: float) -> np.ndarray:
        # screw_axis = [w1, w2, w3, v1, v2, v3]
        w = screw_axis[:3]
        v = screw_axis[3:]
        w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        r = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * (w_hat @ w_hat)
        p = (np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * (w_hat @ w_hat)) @ v
        t = np.eye(4)
        t[:3, :3] = r
        t[:3, 3] = p
        return t
