"""Engine 37: Rayleigh-Plesset Bubble Dynamics + Sonoluminescence Reactor."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class BubbleState:
    radius_m: float = 5.0e-6 # 5 microns
    radial_velocity_m_s: float = 0.0
    core_temp_k: float = 300.0
    photon_emission_flux: float = 0.0

class SonoluminescenceReactorEngine:
    def __init__(self, acoustic_drive_freq_khz: float = 28.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.freq_rad = 2 * np.pi * acoustic_drive_freq_khz * 1e3
        self.state = BubbleState()
        self.r0 = 5.0e-6
        self.p0 = 1.013e5 # 1 atm
        self.rho = 1000.0 # water density
        self.gamma = 1.4 # adiabatic index

    def step_cavitation(self, acoustic_pressure_kpa: float = 135.0, dt_ns: float = 0.1) -> Dict[str, float]:
        """Rayleigh-Plesset equation: R*R'' + 3/2*(R')^2 = (1/rho)*(P_g - P_inf - P_ac)."""
        dt = dt_ns * 1e-9
        r = max(1.0e-7, self.state.radius_m)
        r_dot = self.state.radial_velocity_m_s
        
        p_gas = self.p0 * ((self.r0 / r) ** (3 * self.gamma))
        p_acoustic = acoustic_pressure_kpa * 1e3 * np.sin(self.freq_rad * 1e-6)
        
        # Acceleration
        r_ddot = (1.0 / (self.rho * r)) * (p_gas - self.p0 - p_acoustic) - 1.5 * (r_dot ** 2) / r
        
        self.state.radial_velocity_m_s += r_ddot * dt
        self.state.radius_m += self.state.radial_velocity_m_s * dt

        # Adiabatic core heating upon collapse
        self.state.core_temp_k = float(300.0 * ((self.r0 / max(r, 1e-7)) ** (3 * (self.gamma - 1))))
        self.state.core_temp_k = float(np.clip(self.state.core_temp_k, 300.0, 25000.0))
        
        # Sonoluminescent flash if T > 5000 K (blackbody radiation)
        self.state.photon_emission_flux = float(5.67e-8 * (self.state.core_temp_k ** 4) * 4 * np.pi * (r ** 2))

        return {
            "bubble_radius_um": float(self.state.radius_m * 1e6),
            "core_temperature_k": self.state.core_temp_k,
            "photon_emission_watts": self.state.photon_emission_flux
        }
