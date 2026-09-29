"""Engine 50: Turing Reaction-Diffusion + Morphogenetic Swarm Crystallization."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class SwarmRobot:
    robot_id: int
    pos: np.ndarray # [x, y]
    state_crystallized: bool = False
    cluster_id: int = -1

class MorphogeneticSwarmCrystallizerEngine:
    def __init__(self, num_robots: int = 24, grid_size: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.size = grid_size
        self.num_robots = num_robots
        # Turing Activator (u) and Inhibitor (v) morphogen concentration grids
        self.u = np.full((grid_size, grid_size), 1.0) + self.rng.normal(0, 0.05, size=(grid_size, grid_size))
        self.v = np.full((grid_size, grid_size), 0.5) + self.rng.normal(0, 0.05, size=(grid_size, grid_size))
        # Swarm robots
        self.robots = [
            SwarmRobot(robot_id=i, pos=self.rng.uniform(0.0, float(grid_size - 1), size=2))
            for i in range(num_robots)
        ]

    def step_morphogenesis_and_crystallization(self, dt: float = 0.1) -> Dict[str, float]:
        """Turing PDE (FitzHugh-Nagumo): du/dt = D_u*lap(u) + u - u^3 - v, dv/dt = D_v*lap(v) + gamma*(u - v)."""
        lap_u = (np.roll(self.u, 1, 0) + np.roll(self.u, -1, 0) + np.roll(self.u, 1, 1) + np.roll(self.u, -1, 1) - 4*self.u)
        lap_v = (np.roll(self.v, 1, 0) + np.roll(self.v, -1, 0) + np.roll(self.v, 1, 1) + np.roll(self.v, -1, 1) - 4*self.v)

        du = (0.2 * lap_u + (self.u - self.u**3 - self.v + 0.1)) * dt
        dv = (0.8 * lap_v + 0.5 * (self.u - self.v)) * dt

        self.u = np.clip(self.u + du, -2.0, 2.0)
        self.v = np.clip(self.v + dv, -2.0, 2.0)

        # Swarm crystallization: robots navigate to morphogen peaks (u > 0.8) and lock in place
        crystallized_count = 0
        for r in self.robots:
            gx = int(np.clip(r.pos[0], 0, self.size - 1))
            gy = int(np.clip(r.pos[1], 0, self.size - 1))
            local_morphogen = self.u[gx, gy]
            
            if local_morphogen > 0.5:
                r.state_crystallized = True
                crystallized_count += 1
            else:
                r.state_crystallized = False
                # Drift towards nearest grid neighbor
                r.pos += self.rng.normal(0, 0.2, size=2)
                r.pos = np.clip(r.pos, 0.0, float(self.size - 1))

        return {
            "crystallized_robots": float(crystallized_count),
            "crystallization_fraction": float(crystallized_count / self.num_robots),
            "morphogen_activator_entropy": float(np.var(self.u))
        }
