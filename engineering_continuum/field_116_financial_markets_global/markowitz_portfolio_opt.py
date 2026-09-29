"""Course 116: Markowitz Mean-Variance Portfolio Optimization & Efficient Frontier"""
import numpy as np

class MarkowitzPortfolioOptimizer:
    @staticmethod
    def portfolio_performance(weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> tuple:
        ret = np.dot(weights, expected_returns)
        var = weights.T @ cov_matrix @ weights
        std = np.sqrt(var)
        return float(ret), float(std)

    @staticmethod
    def min_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(cov_matrix.shape[0])
        w = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        return w
