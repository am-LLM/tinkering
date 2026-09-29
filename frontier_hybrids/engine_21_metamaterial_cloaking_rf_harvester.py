"""Engine 21: Metamaterial Transformation Optics + RF Energy Harvester."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple

@dataclass
class CloakHarvestState:
    inner_radius_m: float = 0.05
    outer_radius_m: float = 0.15
    incident_rf_power_mw_cm2: float = 1.2
    harvested_power_mw: float = 0.0
    scattering_cross_section_db: float = -35.0

class MetamaterialCloakRFHarvesterEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = CloakHarvestState()

    def transformation_tensors(self, r: float) -> Tuple[float, float, float]:
        """Compute coordinate transformation tensors: eps_r = mu_r = (r - a)/r, eps_theta = r/(r - a)."""
        a = self.state.inner_radius_m
        b = self.state.outer_radius_m
        if r <= a or r > b:
            return 1.0, 1.0, 1.0
        scale = (b / (b - a))
        eps_r = scale * ((r - a) / r)
        eps_theta = scale * (r / (r - a))
        eps_z = scale * ((b / (b - a)) ** 2) * ((r - a) / r)
        return float(eps_r), float(eps_theta), float(eps_z)

    def step_harvest(self, rf_frequency_ghz: float = 5.8) -> Dict[str, float]:
        """Harvest evanescent wave energy at the inner cloaked boundary via rectenna resonance."""
        # Rectenna conversion efficiency
        rectenna_eff = 0.65
        capture_area_cm2 = 4 * np.pi * ((self.state.inner_radius_m * 100) ** 2)
        total_intercepted = self.state.incident_rf_power_mw_cm2 * capture_area_cm2 * 0.15
        self.state.harvested_power_mw = float(total_intercepted * rectenna_eff)

        return {
            "harvested_power_mw": self.state.harvested_power_mw,
            "scattering_cross_section_db": self.state.scattering_cross_section_db,
            "rectenna_efficiency": rectenna_eff
        }
