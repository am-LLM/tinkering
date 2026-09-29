"""Course 102: Multi-Variable Ordinary Least Squares (OLS) Regression Engine"""
import numpy as np

class OLSEconometricModel:
    def __init__(self):
        self.beta = None
        self.r_squared = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        # Add intercept column
        n = x.shape[0]
        x_design = np.hstack([np.ones((n, 1)), x])
        self.beta = np.linalg.inv(x_design.T @ x_design) @ (x_design.T @ y)
        
        y_pred = x_design @ self.beta
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        self.r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
        return self.beta
