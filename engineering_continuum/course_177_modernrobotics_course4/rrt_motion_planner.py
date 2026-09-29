"""Course 177: Rapidly-exploring Random Tree (RRT) Obstacle-Avoiding Motion Planner"""
import numpy as np

class RRTPlanner:
    def __init__(self, start: tuple, goal: tuple, step_size: float = 1.0):
        self.tree = [start]
        self.parents = {start: None}
        self.goal = goal
        self.step_size = step_size

    def step(self, random_point: tuple) -> tuple:
        # Find nearest
        nearest = min(self.tree, key=lambda p: np.hypot(p[0]-random_point[0], p[1]-random_point[1]))
        dx = random_point[0] - nearest[0]
        dy = random_point[1] - nearest[1]
        dist = np.hypot(dx, dy)
        if dist == 0:
            return nearest
        new_pt = (nearest[0] + self.step_size * dx / dist, nearest[1] + self.step_size * dy / dist)
        self.tree.append(new_pt)
        self.parents[new_pt] = nearest
        return new_pt
