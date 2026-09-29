"""Course 150: TransE Knowledge Graph Embedding Score Function & Link Prediction"""
import numpy as np

class TransEEmbedding:
    def __init__(self, embedding_dim: int = 16, margin: float = 1.0):
        self.dim = embedding_dim
        self.gamma = margin

    @staticmethod
    def score_triple(head: np.ndarray, relation: np.ndarray, tail: np.ndarray, norm_p: int = 2) -> float:
        # Energy function: ||h + r - t||_p
        diff = head + relation - tail
        if norm_p == 1:
            return float(np.sum(np.abs(diff)))
        return float(np.linalg.norm(diff))
