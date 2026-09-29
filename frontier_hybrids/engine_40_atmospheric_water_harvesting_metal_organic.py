"""Engine 40: MOF-801 Hydrophilic Sorption + Solar Atmospheric Water Harvester."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class AtmosphericWaterMOFEngine:
    def __init__(self, mof_mass_kg: float = 2.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.mof_mass = mof_mass_kg
        self.water_uptake_g_g = 0.0 # g H2O / g MOF
        self.total_harvested_liters = 0.0
        self.q_sat = 0.45 # 0.45 g/g saturation uptake for MOF-801

    def sorption_isotherm(self, relative_humidity_pct: float) -> float:
        """Type IV S-shaped isotherm for Zr-MOF (MOF-801): Q(RH) = Q_sat * (K*RH)^n / (1 + (K*RH)^n)."""
        rh = relative_humidity_pct / 100.0
        k = 5.0
        n = 4.0
        uptake = self.q_sat * ((k * rh) ** n) / (1.0 + (k * rh) ** n)
        return float(uptake)

    def step_harvest_cycle(self, ambient_rh_pct: float, solar_irradiance_w_m2: float, dt_hours: float = 1.0) -> Dict[str, float]:
        """Night-time sorption (adsorption) and day-time solar thermal desorption."""
        if solar_irradiance_w_m2 < 100.0:
            # Adsorption phase
            equilibrium_uptake = self.sorption_isotherm(ambient_rh_pct)
            self.water_uptake_g_g += (equilibrium_uptake - self.water_uptake_g_g) * 0.3 * dt_hours
            desorbed_liters = 0.0
        else:
            # Solar thermal desorption phase (condenser collects liquid water)
            desorption_rate = 0.4 * (solar_irradiance_w_m2 / 800.0) * dt_hours
            released_uptake = min(self.water_uptake_g_g, desorption_rate * self.water_uptake_g_g)
            self.water_uptake_g_g -= released_uptake
            desorbed_liters = (released_uptake * self.mof_mass * 1000.0) / 1000.0 # kg = Liters
            self.total_harvested_liters += desorbed_liters

        return {
            "current_uptake_g_per_g": float(self.water_uptake_g_g),
            "desorbed_liters_step": float(desorbed_liters),
            "cumulative_harvested_liters": float(self.total_harvested_liters)
        }
