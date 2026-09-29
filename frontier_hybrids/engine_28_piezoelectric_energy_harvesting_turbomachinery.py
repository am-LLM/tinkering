"""Engine 28: Piezoelectric Vibration Harvesting + sCO2 Turbomachinery Monitor."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class PiezoTurbomachineryEngine:
    def __init__(self, blade_pass_freq_hz: float = 1200.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.bpf = blade_pass_freq_hz
        self.stored_energy_uj = 0.0
        self.piezo_d33 = 450e-12 # C/N piezoelectric charge constant

    def step_vibration_harvest(self, vibration_amplitude_g: float = 8.5, duration_ms: float = 10.0) -> Dict[str, float]:
        """Harvest kinetic vibration power: P = 0.5 * C * V^2 * f."""
        # Acceleration to dynamic inertial force on bimorph tip mass
        tip_mass_kg = 0.005 # 5g
        inertial_force_n = tip_mass_kg * vibration_amplitude_g * 9.81
        
        # Generated open-circuit voltage
        v_oc = inertial_force_n * 12.0 # Volts
        capacitance = 25e-9 # 25 nF
        
        harvested_power_mw = 0.5 * capacitance * (v_oc ** 2) * self.bpf * 1e3
        d_energy_uj = harvested_power_mw * duration_ms
        self.stored_energy_uj += d_energy_uj

        # Sensor transmission trigger threshold: 50 uJ
        packet_transmitted = self.stored_energy_uj >= 50.0
        if packet_transmitted:
            self.stored_energy_uj -= 50.0

        return {
            "harvested_power_mw": float(harvested_power_mw),
            "stored_energy_uj": float(self.stored_energy_uj),
            "packet_transmitted": 1.0 if packet_transmitted else 0.0
        }
