"""Engine 38: Terfenol-D Magnetostriction + Deep-Sea Broadband Sonar."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class MagnetostrictiveSonarEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Giant magnetostrictive coefficient d33 = 1.6e-8 m/A
        self.d33 = 1.6e-8
        self.rod_length_m = 0.15
        self.youngs_modulus_gpa = 30.0

    def step_transmit_pulse(self, coil_current_a: float = 12.0, water_depth_m: float = 2500.0) -> Dict[str, float]:
        """Piezomagnetic constitutive strain: S = d33 * H + s^H * T."""
        h_field_a_m = coil_current_a * 1000.0 # Solenoid turns density
        # Magnetostrictive longitudinal displacement
        strain = self.d33 * h_field_a_m
        displacement_um = strain * self.rod_length_m * 1e6

        # Radiated acoustic pressure P = rho * c * v
        radiated_pressure_kpa = 1025.0 * 1500.0 * (displacement_um * 1e-6 * 2000.0) * 1e-3
        # Source level SL in dB re 1 uPa @ 1m
        source_level_db = 20.0 * np.log10(max(1.0, radiated_pressure_kpa * 1e9))

        return {
            "rod_displacement_um": float(displacement_um),
            "radiated_pressure_kpa": float(radiated_pressure_kpa),
            "source_level_db": float(source_level_db),
            "hydrostatic_pressure_mpa": float(1025.0 * 9.81 * water_depth_m * 1e-6)
        }
