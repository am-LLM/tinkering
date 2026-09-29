"""Course 232: 2D Log-Odds Bayesian Occupancy Grid Mapping SLAM Engine"""
import numpy as np

class OccupancyGridSLAM:
    def __init__(self, width: int = 20, height: int = 20, l_occ: float = 0.85, l_free: float = -0.4):
        self.grid = np.zeros((width, height), dtype=float)
        self.l_occ = l_occ
        self.l_free = l_free

    def update_ray(self, robot_x: int, robot_y: int, hit_x: int, hit_y: int):
        self.grid[hit_x, hit_y] += self.l_occ
        # Free space along mid-point
        mid_x = (robot_x + hit_x) // 2
        mid_y = (robot_y + hit_y) // 2
        self.grid[mid_x, mid_y] += self.l_free

    def get_probability_grid(self) -> np.ndarray:
        return 1.0 - (1.0 / (1.0 + np.exp(self.grid)))
