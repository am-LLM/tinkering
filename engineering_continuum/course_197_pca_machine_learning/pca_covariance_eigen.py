"""Course 197: PCA via Covariance Eigendecomposition & Variance Ratios"""
import numpy as np

class PCACovarianceEigen:
    @staticmethod
    def compute_pca(x: np.ndarray, n_components: int = 2) -> tuple:
        x_centered = x - np.mean(x, axis=0)
        cov = np.cov(x_centered, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvals)[::-1]
        sorted_vals = eigenvals[idx]
        sorted_vecs = eigenvecs[:, idx]
        
        components = sorted_vecs[:, :n_components]
        projected = x_centered @ components
        var_ratios = sorted_vals[:n_components] / np.sum(sorted_vals)
        return projected, var_ratios
