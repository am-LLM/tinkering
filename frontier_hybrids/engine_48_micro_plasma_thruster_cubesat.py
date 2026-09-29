"""Engine 48: Helicon Plasma RF Wave + CubeSat Thrust Vector Control."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class ThrusterState:
    rf_power_w: float = 50.0
    mass_flow_mg_s: float = 0.2 # Xenon propellant
    exhaust_velocity_km_s: float = 0.0
    thrust_mn: float = 0.0
    specific_impulse_sec: float = 0.0

class HeliconPlasmaThrusterEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = ThrusterState()
        self.xenon_ion_mass_kg = 2.18e-25
        self.q_elem = 1.602e-19

    def step_thrust_vector(self, magnetic_nozzle_angle_deg: float) -> Dict[str, float]:
        """Compute plasma ionization, magnetic nozzle acceleration, and vector thrust."""
        # Plasma potential drop V_p = 0.5 * (RF_power / mass_flow)
        v_plasma_drop = 300.0 * (self.state.rf_power_w / 50.0) / (self.state.mass_flow_mg_s / 0.2)
        
        # Ion exhaust velocity: v_ex = sqrt(2 * q * V_p / M_ion)
        v_ex_m_s = np.sqrt(2 * self.q_elem * v_plasma_drop / self.xenon_ion_mass_kg)
        self.state.exhaust_velocity_km_s = float(v_ex_m_s * 1e-3)
        
        # Thrust F = m_dot * v_ex
        mass_flow_kg_s = self.state.mass_flow_mg_s * 1e-6
        total_thrust_n = mass_flow_kg_s * v_ex_m_s
        self.state.thrust_mn = float(total_thrust_n * 1e3)
        self.state.specific_impulse_sec = float(v_ex_m_s / 9.81)

        # Vector decomposition
        rad = np.radians(magnetic_nozzle_angle_deg)
        f_axial_mn = self.state.thrust_mn * np.cos(rad)
        f_transverse_mn = self.state.thrust_mn * np.sin(rad)

        return {
            "total_thrust_mn": self.state.thrust_mn,
            "specific_impulse_sec": self.state.specific_impulse_sec,
            "axial_thrust_mn": float(f_axial_mn),
            "transverse_thrust_mn": float(f_transverse_mn),
            "exhaust_velocity_km_s": self.state.exhaust_velocity_km_s
        }
