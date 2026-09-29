"""
Engine 56: Suprachiasmatic Nucleus Circadian Clock + Microgrid Battery Thermal Health.
"""
import numpy as np

class CircadianMicrogridManager:
    def __init__(self, battery_capacity_kwh: float = 1000.0):
        self.capacity = battery_capacity_kwh
        self.soc = 0.65

    def step_circadian_shedding(self, hour_of_day: float, load_demand_kw: float, solar_gen_kw: float) -> dict:
        # Cortisol / Melatonin endocrine phase curve
        cortisol_phase = np.exp(-((hour_of_day - 9.0) ** 2) / 18.0)
        melatonin_phase = np.exp(-((hour_of_day - 23.0) % 24) ** 2 / 12.0)
        
        priority_load_shed = max(0.0, load_demand_kw * (1.0 - 0.3 * melatonin_phase))
        net_power = solar_gen_kw - priority_load_shed
        self.soc = float(np.clip(self.soc + (net_power / self.capacity) * 0.05, 0.1, 0.95))
        
        return {
            "dispatched_load_kw": float(priority_load_shed),
            "battery_soc": self.soc,
            "circadian_phase_factor": float(cortisol_phase - melatonin_phase)
        }
