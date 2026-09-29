"""Course 160: Steinmetz High-Frequency Magnetic Core Loss & Skin Depth Calculator"""
import math

class SteinmetzCoreLoss:
    @staticmethod
    def core_loss_per_volume(frequency_hz: float, b_peak_tesla: float, k: float = 1.5, alpha: float = 1.3, beta: float = 2.4) -> float:
        # P_v = k * f^alpha * B_peak^beta (W/m^3)
        return float(k * (frequency_hz ** alpha) * (b_peak_tesla ** beta))

    @staticmethod
    def skin_depth_copper(frequency_hz: float, resistivity: float = 1.68e-8, mu_r: float = 1.0) -> float:
        # delta = sqrt(rho / (pi * f * mu))
        mu = mu_r * 4.0 * math.pi * 1e-7
        return float(math.sqrt(resistivity / (math.pi * frequency_hz * mu)))
