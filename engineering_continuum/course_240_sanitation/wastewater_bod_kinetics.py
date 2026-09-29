"""Course 240: Wastewater Biological Oxygen Demand (BOD5) First-Order Degradation Kinetics"""
import math

class WastewaterBODKinetics:
    @staticmethod
    def bod_t(bod_ultimate: float, k_decay_day: float, t_days: float) -> float:
        # BOD(t) = L0 * (1 - exp(-k * t))
        return float(bod_ultimate * (1.0 - math.exp(-k_decay_day * t_days)))

    @staticmethod
    def clarifier_settling_velocity(particle_density: float, water_density: float, particle_diam_m: float, dynamic_viscosity: float, g: float = 9.81) -> float:
        # Stokes law settling velocity: v_s = g*(rho_p - rho_w)*d^2 / (18 * mu)
        return float((g * (particle_density - water_density) * (particle_diam_m ** 2)) / (18.0 * dynamic_viscosity))
