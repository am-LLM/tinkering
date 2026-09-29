"""Course 172: Schelling Spatial Segregation Cellular Automaton Simulator"""
import numpy as np

class SchellingSegregationSim:
    def __init__(self, grid_size: int = 10, similarity_threshold: float = 0.3):
        self.size = grid_size
        self.threshold = similarity_threshold
        # 0: Empty, 1: Red, 2: Blue
        self.grid = np.random.choice([0, 1, 2], size=(grid_size, grid_size), p=[0.2, 0.4, 0.4])

    def satisfaction_rate(self) -> float:
        satisfied = 0
        agents = 0
        for i in range(self.size):
            for j in range(self.size):
                agent = self.grid[i, j]
                if agent == 0:
                    continue
                agents += 1
                neighbors = self.grid[max(0, i-1):min(self.size, i+2), max(0, j-1):min(self.size, j+2)]
                similar = np.sum(neighbors == agent) - 1
                total_neighbors = np.sum(neighbors != 0) - 1
                if total_neighbors == 0 or (similar / total_neighbors) >= self.threshold:
                    satisfied += 1
        return float(satisfied / agents) if agents > 0 else 1.0
