"""Engine 13: Thermoacoustic Wave Oscillation + Pulse-Tube Stirling Cryocooler."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class CryocoolerState:
    hot_temp_k: float = 300.0
    cold_temp_k: float = 77.0
    acoustic_pressure_amplitude_kpa: float = 250.0
    frequency_hz: float = 60.0
    phase_angle_rad: float = 0.52 # ~30 deg between pressure & velocity
    carnot_efficiency: float = 0.0

class ThermoacousticCryocoolerEngine:
    def __init__(self, charge_pressure_mpa: float = 3.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.charge_pressure = charge_pressure_mpa
        self.state = CryocoolerState()

    def step_cryocooling(self, heat_load_watts: float = 5.0, dt: float = 0.1) -> Dict[str, float]:
        """Compute Rott acoustic power and Enthalpy flow: H_dot = 0.5 * Re(P1 * U1*) - k * dT/dx."""
        p1 = self.state.acoustic_pressure_amplitude_kpa * 1e3
        u1_volumetric = 0.002 # m^3/s peak velocity
        
        # Acoustic work flow W_dot = 0.5 * |P1| * |U1| * cos(phase)
        acoustic_work = 0.5 * p1 * u1_volumetric * np.cos(self.state.phase_angle_rad)
        
        # Cooling capacity at cold end
        cop_carnot = self.state.cold_temp_k / (self.state.hot_temp_k - self.state.cold_temp_k)
        ideal_cooling = acoustic_work * cop_carnot * 0.15 # 15% of Carnot
        
        net_cooling_power = ideal_cooling - heat_load_watts
        
        # Temperature evolution: dT/dt = -net_cooling / thermal_capacity
        thermal_mass = 50.0 # J/K
        self.state.cold_temp_k -= (net_cooling_power / thermal_mass) * dt
        self.state.cold_temp_k = float(np.clip(self.state.cold_temp_k, 4.2, self.state.hot_temp_k))
        self.state.carnot_efficiency = float(cop_carnot)

        return {
            "cold_temp_k": self.state.cold_temp_k,
            "acoustic_work_watts": float(acoustic_work),
            "net_cooling_power_watts": float(net_cooling_power),
            "cop_carnot": self.state.carnot_efficiency
        }
