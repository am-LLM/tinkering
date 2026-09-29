"""Course 242: Squeezed Light State Evolution & Wigner Distribution Engine"""
import numpy as np

class SqueezedStateEngine:
    def __init__(self, squeeze_param_r: float = 0.8, squeeze_angle_theta: float = 0.0):
        self.r = squeeze_param_r
        self.theta = squeeze_angle_theta

    def quadrature_variances(self) -> tuple[float, float]:
        var_x1 = 0.5 * np.exp(-2.0 * self.r)
        var_x2 = 0.5 * np.exp(2.0 * self.r)
        return float(var_x1), float(var_x2)

    def wigner_function_2d(self, x_grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
        var_x, var_p = self.quadrature_variances()
        X, P = np.meshgrid(x_grid, p_grid)
        w = (1.0 / (2.0 * np.pi * np.sqrt(var_x * var_p))) * np.exp(- (X**2 / (2.0 * var_x) + P**2 / (2.0 * var_p)))
        return w
