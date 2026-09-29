"""Course 052: 2D Plane Stress Transformation & Mohr's Circle Principal Tensor Solver"""
import numpy as np

class MohrCircleStress:
    @staticmethod
    def principal_stresses(sigma_x: float, sigma_y: float, tau_xy: float):
        sigma_avg = (sigma_x + sigma_y) / 2.0
        r = np.sqrt(((sigma_x - sigma_y) / 2.0) ** 2 + tau_xy ** 2)
        sigma_1 = sigma_avg + r
        sigma_2 = sigma_avg - r
        tau_max = r
        theta_p = 0.5 * np.arctan2(2.0 * tau_xy, sigma_x - sigma_y)
        return {
            "sigma_1": float(sigma_1),
            "sigma_2": float(sigma_2),
            "tau_max": float(tau_max),
            "theta_p_deg": float(np.degrees(theta_p))
        }
