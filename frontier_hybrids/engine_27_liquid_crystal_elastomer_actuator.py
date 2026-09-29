"""Engine 27: Frank Elastic Director Nematodynamics + LCE Peristaltic Crawler."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class LiquidCrystalElastomerEngine:
    def __init__(self, num_nodes: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        # Nematic order parameter S and director angle theta
        self.director_angle = np.zeros(num_nodes)
        self.nematic_order_s = np.full(num_nodes, 0.6)
        self.body_displacement = 0.0

    def step_peristaltic_crawl(self, thermal_stimulus_phase: float, dt: float = 0.05) -> Dict[str, float]:
        """Thermal stimulus alters nematic order S(T) -> induces uniaxial mechanical strain."""
        for i in range(self.num_nodes):
            phase = thermal_stimulus_phase + 2 * np.pi * (i / self.num_nodes)
            # Local heating drops order parameter S
            self.nematic_order_s[i] = float(0.6 - 0.3 * (0.5 * (np.sin(phase) + 1.0)))
            self.director_angle[i] = float(0.2 * np.cos(phase))

        # Contraction strain: eps = lambda_0 * (S - S_0)
        local_strains = 0.4 * (self.nematic_order_s - 0.6)
        # Asymmetric ratchet friction produces net forward crawl
        net_advance = float(-np.sum(local_strains * np.sign(np.gradient(local_strains)))) * 0.05
        self.body_displacement += net_advance

        return {
            "total_body_displacement_mm": float(self.body_displacement * 1e3),
            "mean_nematic_order": float(np.mean(self.nematic_order_s)),
            "max_strain": float(np.max(abs(local_strains)))
        }
