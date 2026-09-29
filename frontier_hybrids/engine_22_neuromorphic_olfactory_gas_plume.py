"""Engine 22: Neuromorphic Olfactory Glomeruli + UAV Gas Plume Tracking."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class UAVState:
    pos: np.ndarray # [x, y]
    heading_rad: float # orientation
    speed: float = 2.0 # m/s
    state_mode: str = "SEARCH" # SEARCH, SURGE, CAST

class NeuromorphicOlfactoryPlumeEngine:
    def __init__(self, num_glomeruli: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_glomeruli = num_glomeruli
        self.uav = UAVState(pos=np.array([0.0, 0.0]), heading_rad=0.0)
        self.plume_source = np.array([25.0, 15.0])
        self.glomerular_weights = self.rng.uniform(0.5, 1.5, size=num_glomeruli)

    def sample_chemical_concentration(self, pos: np.ndarray) -> float:
        """Gaussian filament gas plume model."""
        dist = np.linalg.norm(pos - self.plume_source)
        # Intermittent turbulent odor packet encounters
        turbulent_pulse = self.rng.exponential(1.0) if self.rng.uniform(0, 1) < 0.4 else 0.0
        conc = np.exp(-0.1 * dist) * 10.0 + turbulent_pulse
        return float(max(0.0, conc))

    def step_olfactory_nav(self, dt: float = 0.2) -> Dict[str, float]:
        raw_conc = self.sample_chemical_concentration(self.uav.pos)
        # Glomerular layer contrast enhancement via lateral inhibition
        glom_activity = np.clip(raw_conc * self.glomerular_weights - 0.2 * np.mean(self.glomerular_weights), 0, None)
        perceived_odor = float(np.sum(glom_activity))

        # Moth-inspired surge-and-cast strategy
        if perceived_odor > 2.0:
            self.uav.state_mode = "SURGE"
            # Surge upwind / towards gradient
            angle_to_source = np.arctan2(self.plume_source[1] - self.uav.pos[1], self.plume_source[0] - self.uav.pos[0])
            self.uav.heading_rad = angle_to_source + self.rng.normal(0, 0.1)
        else:
            self.uav.state_mode = "CAST"
            # Crosswind zigzag casting
            self.uav.heading_rad += np.pi / 2.0 * (1.0 if self.rng.uniform(0, 1) > 0.5 else -1.0)

        # Update position
        vel = self.uav.speed * np.array([np.cos(self.uav.heading_rad), np.sin(self.uav.heading_rad)])
        self.uav.pos += vel * dt

        distance_to_source = float(np.linalg.norm(self.uav.pos - self.plume_source))
        return {
            "perceived_odor": perceived_odor,
            "distance_to_source_m": distance_to_source,
            "uav_mode_surge": 1.0 if self.uav.state_mode == "SURGE" else 0.0
        }
