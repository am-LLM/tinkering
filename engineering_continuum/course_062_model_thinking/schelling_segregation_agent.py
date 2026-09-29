"""Course 062: Schelling Spatial Segregation Agent-Based Dynamics Engine"""
import numpy as np

class SchellingGrid:
    def __init__(self, size=20, empty_ratio=0.15, similarity_threshold=0.35):
        self.size = size
        self.threshold = similarity_threshold
        n_cells = size * size
        n_empty = int(n_cells * empty_ratio)
        n_a = (n_cells - n_empty) // 2
        n_b = n_cells - n_empty - n_a
        flat = [0] * n_empty + [1] * n_a + [2] * n_b
        np.random.seed(42)
        np.random.shuffle(flat)
        self.grid = np.array(flat).reshape((size, size))

    def step(self) -> float:
        unhappy = []
        for r in range(self.size):
            for c in range(self.size):
                agent = self.grid[r, c]
                if agent == 0:
                    continue
                neighbors = self.grid[max(0, r-1):min(self.size, r+2), max(0, c-1):min(self.size, c+2)].flatten()
                occupied = neighbors[neighbors != 0]
                if len(occupied) > 1:
                    same = np.sum(occupied == agent) - 1
                    sim = same / (len(occupied) - 1)
                    if sim < self.threshold:
                        unhappy.append((r, c))
                        
        empty_locs = list(zip(*np.where(self.grid == 0)))
        for r, c in unhappy:
            if not empty_locs:
                break
            dest_idx = np.random.randint(len(empty_locs))
            dest = empty_locs.pop(dest_idx)
            self.grid[dest[0], dest[1]] = self.grid[r, c]
            self.grid[r, c] = 0
            empty_locs.append((r, c))
            
        return len(unhappy) / (self.size * self.size)
