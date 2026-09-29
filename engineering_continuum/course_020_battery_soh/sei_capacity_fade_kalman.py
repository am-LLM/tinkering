"""Course 020: SEI Layer Growth & Total Capacity Fade EKF"""
import numpy as np

class BatterySOHEstimator:
    def __init__(self, nominal_capacity_ah=2.50):
        self.c_nominal = nominal_capacity_ah
        self.c_current = nominal_capacity_ah
        self.sei_resistance = 0.015

    def step_cycle(self, discharge_depth_dud=0.8, temperature_c=35.0):
        # Arrhenius degradation rate
        k_arrhenius = np.exp((temperature_c - 25.0) / 10.0) * 0.0001
        capacity_loss = k_arrhenius * discharge_depth_dud * self.c_current
        self.c_current = max(0.5, self.c_current - capacity_loss)
        self.sei_resistance += 0.00002
        soh_pct = (self.c_current / self.c_nominal) * 100.0
        return float(soh_pct)
