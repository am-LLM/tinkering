"""Course 120: Functional Connectivity Correlation Matrix & Graph Centrality"""
import numpy as np

class FMRIConnectivityGraph:
    @staticmethod
    def correlation_matrix(roi_time_series: np.ndarray) -> np.ndarray:
        # roi_time_series shape: (N_rois, T_timepoints)
        return np.corrcoef(roi_time_series)

    @staticmethod
    def binarize_network(corr_matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        adj = np.abs(corr_matrix) > threshold
        np.fill_diagonal(adj, False)
        return adj.astype(int)

    @staticmethod
    def degree_centrality(adjacency_matrix: np.ndarray) -> np.ndarray:
        return np.sum(adjacency_matrix, axis=1)
