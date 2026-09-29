"""Course 044: Carbon Nanotube / Graphene Chemiresistor Gas Adsorption Engine"""
import numpy as np

class NanomaterialChemiresistor:
    def __init__(self, r_baseline=10000.0, k_adsorption=0.05, k_desorption=0.01, sensitivity_coeff=0.25):
        self.r0 = r_baseline
        self.k_ads = k_adsorption
        self.k_des = k_desorption
        self.s = sensitivity_coeff
        self.theta = 0.0 # Surface fractional coverage

    def step_exposure(self, gas_concentration_ppm: float, dt: float = 0.1) -> float:
        # Langmuir kinetics: dTheta/dt = k_ads * C * (1 - Theta) - k_des * Theta
        d_theta = (self.k_ads * gas_concentration_ppm * (1.0 - self.theta) - self.k_des * self.theta) * dt
        self.theta = float(np.clip(self.theta + d_theta, 0.0, 1.0))
        
        # Delta R / R0 = sensitivity * Theta
        r_current = self.r0 * (1.0 + self.s * self.theta)
        return r_current
