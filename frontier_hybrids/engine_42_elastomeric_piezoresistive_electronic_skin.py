"""Engine 42: Percolation Piezoresistive E-Skin + Tactile Slip Detection."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class PiezoresistiveElectronicSkinEngine:
    def __init__(self, num_taxels: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_taxels = num_taxels
        # Percolation volume fraction p vs critical p_c = 0.18
        self.carbon_black_fraction = 0.22
        self.taxel_conductance = np.full(num_taxels, 1.0e-3)
        self.slip_detected = False

    def step_tactile_sense(self, normal_pressure_kpa: float, shear_vibration_hz: float, dt: float = 0.01) -> Dict[str, float]:
        """Piezoresistive percolation: sigma ~ (p - p_c)^t with pressure-induced contact tunneling."""
        # Normal pressure compresses matrix, increasing local particle density p
        effective_p = self.carbon_black_fraction + 0.0005 * normal_pressure_kpa
        base_conductance = 1.0e-2 * ((effective_p - 0.18) ** 1.8)
        
        # High-frequency shear vibration (stick-slip transition ~ 200-400 Hz)
        shear_mod = 0.2 * np.sin(2 * np.pi * shear_vibration_hz * dt) if shear_vibration_hz > 150.0 else 0.0
        self.taxel_conductance = base_conductance * (1.0 + shear_mod + self.rng.normal(0, 0.01, size=self.num_taxels))
        
        # Slip event detection via conductance derivative variance
        conductance_diff = float(np.var(self.taxel_conductance))
        self.slip_detected = bool(conductance_diff > 1.0e-8 and shear_vibration_hz > 180.0)

        return {
            "mean_conductance_ms": float(np.mean(self.taxel_conductance) * 1e3),
            "conductance_variance": conductance_diff,
            "slip_alarm": 1.0 if self.slip_detected else 0.0
        }
