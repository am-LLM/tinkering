"""Engine 26: Hartmann Flow MHD + Molten Salt Nuclear Coolant Pump."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class MHDPumpState:
    channel_width_m: float = 0.05
    magnetic_field_b_tesla: float = 1.2
    applied_current_a: float = 500.0
    fluid_density_kg_m3: float = 1950.0 # FLiNaK molten salt
    electrical_conductivity_s_m: float = 180.0
    flow_velocity_m_s: float = 0.0

class MHDMoltenSaltPumpEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = MHDPumpState()

    def step_pumping(self, loop_pressure_drop_pa: float = 2.0e4, dt: float = 0.01) -> Dict[str, float]:
        """Compute Hartmann profile and electromagnetic Lorentz head: Delta P = (J x B) * L."""
        b = self.state.magnetic_field_b_tesla
        j_density = self.state.applied_current_a / (self.state.channel_width_m ** 2)
        
        # Lorentz driving force F_lorentz = J x B
        lorentz_pressure = j_density * b * self.state.channel_width_m
        
        # Hartmann braking counter-EMF
        back_emf_current = self.state.electrical_conductivity_s_m * (self.state.flow_velocity_m_s * b)
        net_driving_pressure = lorentz_pressure - back_emf_current * b * self.state.channel_width_m
        
        # Acceleration: du/dt = (Delta P_drive - Delta P_loss) / (rho * L)
        net_head = net_driving_pressure - loop_pressure_drop_pa
        accel = net_head / (self.state.fluid_density_kg_m3 * self.state.channel_width_m)
        self.state.flow_velocity_m_s = float(np.clip(self.state.flow_velocity_m_s + accel * dt, 0.0, 15.0))

        volumetric_flow_m3_s = self.state.flow_velocity_m_s * (self.state.channel_width_m ** 2)
        return {
            "flow_velocity_m_s": self.state.flow_velocity_m_s,
            "lorentz_pressure_pa": float(lorentz_pressure),
            "volumetric_flow_lps": float(volumetric_flow_m3_s * 1000.0)
        }
