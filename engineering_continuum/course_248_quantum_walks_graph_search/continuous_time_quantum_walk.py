"""Course 248: Continuous-Time Quantum Walk Graph Search Solver"""
import numpy as np
import scipy.linalg

class ContinuousTimeQuantumWalk:
    def __init__(self, adjacency_matrix: np.ndarray, gamma_hopping: float = 1.0):
        self.adj = np.array(adjacency_matrix, dtype=float)
        self.n = len(self.adj)
        self.deg = np.diag(np.sum(self.adj, axis=1))
        self.laplacian = self.deg - self.adj
        self.gamma = gamma_hopping

    def evolve_state(self, psi_0: np.ndarray, t: float, target_node: int = None, oracle_strength: float = 0.0) -> np.ndarray:
        h = self.gamma * self.laplacian
        if target_node is not None:
            oracle = np.zeros_like(h)
            oracle[target_node, target_node] = -oracle_strength
            h = h + oracle
        u = scipy.linalg.expm(-1j * h * t)
        psi_t = u @ psi_0
        return psi_t

    def probability_distribution(self, psi: np.ndarray) -> np.ndarray:
        return np.abs(psi)**2
