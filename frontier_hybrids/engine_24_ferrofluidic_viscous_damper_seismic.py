"""Engine 24: Ferrofluid Rheology + Tuned Mass Damper Seismic Protection."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class BuildingStructure:
    mass_kg: float = 1.0e5
    stiffness_n_m: float = 5.0e6
    damping_ns_m: float = 5.0e4
    pos_m: float = 0.0
    vel_m_s: float = 0.0

@dataclass
class FerrofluidDamper:
    mass_kg: float = 5.0e3
    magnetic_field_tesla: float = 0.5
    pos_m: float = 0.0
    vel_m_s: float = 0.0

class FerrofluidSeismicEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.bldg = BuildingStructure()
        self.tmd = FerrofluidDamper()

    def step_seismic(self, ground_accel_ms2: float, dt: float = 0.005) -> Dict[str, float]:
        """Compute Rosensweig magnetic viscosity damping: eta = eta_0 * (1 + 1.5 * alpha * L(alpha))."""
        # Langevin parameter alpha = mu_0 * m * H / (k_B * T)
        alpha = self.tmd.magnetic_field_tesla * 10.0
        langevin = (1.0 / np.tanh(alpha + 1e-6)) - (1.0 / (alpha + 1e-6))
        effective_viscosity = 1.0e4 * (1.0 + 3.0 * langevin)

        # Equations of motion for 2-DOF system
        rel_vel = self.tmd.vel_m_s - self.bldg.vel_m_s
        f_damper = effective_viscosity * rel_vel + 5.0e4 * (self.tmd.pos_m - self.bldg.pos_m)

        # Building accel
        a_bldg = (-self.bldg.stiffness_n_m * self.bldg.pos_m - self.bldg.damping_ns_m * self.bldg.vel_m_s + f_damper) / self.bldg.mass_kg - ground_accel_ms2
        # TMD accel
        a_tmd = (-f_damper) / self.tmd.mass_kg - ground_accel_ms2

        self.bldg.vel_m_s += a_bldg * dt
        self.bldg.pos_m += self.bldg.vel_m_s * dt
        self.tmd.vel_m_s += a_tmd * dt
        self.tmd.pos_m += self.tmd.vel_m_s * dt

        return {
            "building_displacement_mm": float(self.bldg.pos_m * 1e3),
            "effective_viscosity": float(effective_viscosity),
            "damper_force_kn": float(f_damper * 1e-3)
        }
