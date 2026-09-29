"""Course 216: PageRank Power Iteration Graph Engine with Teleportation Damping"""
import numpy as np

class PageRankGraphEngine:
    @staticmethod
    def compute_pagerank(adj_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 50) -> np.ndarray:
        n = adj_matrix.shape[0]
        # Normalize columns (out-degree)
        col_sums = np.sum(adj_matrix, axis=0)
        col_sums[col_sums == 0] = 1.0
        m = adj_matrix / col_sums
        
        pr = np.ones(n) / n
        for _ in range(max_iter):
            pr = (1.0 - damping) / n + damping * (m @ pr)
        return pr
