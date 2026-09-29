"""Engine 18: Laser-Plasma Wakefield Accelerator + Beam Quality Emittance."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class WakefieldState:
    laser_a0: float = 3.5 # Normalized vector potential
    plasma_density_cm3: float = 1.5e18
    beam_energy_gev: float = 0.0
    energy_spread_percent: float = 5.0
    transverse_emittance_mm_mrad: float = 0.8

class LaserPlasmaWakefieldEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = WakefieldState()

    def step_acceleration(self, propagation_distance_cm: float = 2.0) -> Dict[str, float]:
        """Compute accelerating field: E_z = E_0 * sqrt(a_0) where E_0 = 96 * sqrt(n_0[cm^-3]) V/m."""
        e0_v_m = 96.0 * np.sqrt(self.state.plasma_density_cm3)
        e_accel_gv_m = (e0_v_m * np.sqrt(self.state.laser_a0)) * 1e-9
        
        # Energy gain = E_accel * L
        d_energy = e_accel_gv_m * (propagation_distance_cm * 1e-2)
        self.state.beam_energy_gev += float(d_energy)

        # Dephasing & beam loading effects
        self.state.energy_spread_percent = float(np.clip(2.0 + 0.5 * np.log1p(self.state.beam_energy_gev), 0.5, 10.0))
        self.state.transverse_emittance_mm_mrad = float(0.8 / (1.0 + 0.1 * self.state.beam_energy_gev))

        return {
            "beam_energy_gev": self.state.beam_energy_gev,
            "accelerating_gradient_gv_m": float(e_accel_gv_m),
            "energy_spread_percent": self.state.energy_spread_percent,
            "transverse_emittance_mm_mrad": self.state.transverse_emittance_mm_mrad
        }
