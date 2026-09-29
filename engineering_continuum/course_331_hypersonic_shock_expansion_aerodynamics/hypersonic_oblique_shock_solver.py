"""Course 331: Hypersonic Oblique Shock and Prandtl-Meyer Expansion Solver"""
import numpy as np

class HypersonicShockSolver:
    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    def oblique_shock_pressure_ratio(self, mach_inf: float, shock_wave_angle_rad: float) -> float:
        # P2 / P1 across oblique shock
        m1 = mach_inf
        beta = shock_wave_angle_rad
        mn1 = m1 * np.sin(beta)
        if mn1 < 1.0:
            return 1.0
        p_ratio = 1.0 + (2.0 * self.gamma / (self.gamma + 1.0)) * (mn1**2 - 1.0)
        return float(p_ratio)

    def newtonian_impact_pressure_coefficient(self, deflection_angle_rad: float) -> float:
        # Modified Newtonian impact theory for high Mach numbers: Cp = 2 * sin^2(theta)
        return float(2.0 * (np.sin(deflection_angle_rad)**2))
