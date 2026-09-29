"""Course 206: Gaussian Maximum Likelihood (MLE) & MAP Parameter Estimation"""
import numpy as np

class GaussianMLEMAP:
    @staticmethod
    def mle_estimates(data: np.ndarray) -> tuple:
        mu_mle = float(np.mean(data))
        sigma2_mle = float(np.mean((data - mu_mle)**2))
        return mu_mle, sigma2_mle

    @staticmethod
    def map_mean(data: np.ndarray, prior_mu: float, prior_var: float, known_var: float) -> float:
        n = len(data)
        data_mean = np.mean(data)
        post_mean = (prior_mu / prior_var + (n * data_mean) / known_var) / (1.0 / prior_var + n / known_var)
        return float(post_mean)
