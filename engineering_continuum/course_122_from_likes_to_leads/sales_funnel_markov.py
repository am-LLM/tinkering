"""Course 122: Markov Chain Customer Conversion Funnel Transition Matrix"""
import numpy as np

class SalesFunnelMarkov:
    def __init__(self, transition_matrix: np.ndarray, state_names: list):
        self.t = transition_matrix
        self.states = state_names

    def simulate_absorption(self, initial_dist: np.ndarray, steps: int = 10) -> np.ndarray:
        dist = initial_dist
        for _ in range(steps):
            dist = dist @ self.t
        return dist
