"""Engine 46: Seebeck Thermoelectricity + Deep Space Radioisotope Generator."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class ThermoelectricRTGEngine:
    def __init__(self, initial_thermal_power_w: float = 2000.0, isotope_half_life_years: float = 87.7, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.p_th0 = initial_thermal_power_w # Pu-238 heat source
        self.half_life_yr = isotope_half_life_years
        # SiGe thermocouple properties
        self.seebeck_coeff_uv_k = 280.0 # 280 uV/K
        self.num_couples = 312
        self.internal_resistance_ohm = 1.2

    def step_mission_years(self, mission_elapsed_years: float, space_sink_temp_k: float = 4.0) -> Dict[str, float]:
        """Radioactive decay P_th(t) = P_0 * 2^(-t/t_1/2) and Seebeck voltage V = N * S * Delta T."""
        current_p_th = self.p_th0 * (0.5 ** (mission_elapsed_years / self.half_life_yr))
        
        # Hot junction equilibrium temperature
        t_hot_k = space_sink_temp_k + (current_p_th / 2.5) # Thermal conductance ~ 2.5 W/K
        delta_t = t_hot_k - space_sink_temp_k
        
        # Seebeck open-circuit voltage
        v_oc = self.num_couples * (self.seebeck_coeff_uv_k * 1e-6) * delta_t
        
        # Max power transfer to matched load (R_load = R_int)
        p_electric = (v_oc ** 2) / (4 * self.internal_resistance_ohm)
        rtg_efficiency_pct = (p_electric / current_p_th) * 100.0

        return {
            "thermal_power_w": float(current_p_th),
            "electrical_power_w": float(p_electric),
            "open_circuit_voltage_v": float(v_oc),
            "rtg_efficiency_pct": float(rtg_efficiency_pct),
            "hot_junction_temp_k": float(t_hot_k)
        }
