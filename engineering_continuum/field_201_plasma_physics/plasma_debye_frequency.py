"""Course 201: Debye Shielding Length & Electron Plasma Frequency Calculator"""
import math

class PlasmaDebyeFrequency:
    EPSILON_0 = 8.854187817e-12
    Q_E = 1.602176634e-19
    M_E = 9.1093837e-31
    K_B = 1.380649e-23

    @classmethod
    def plasma_frequency_rad_s(cls, n_e_m3: float) -> float:
        # omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))
        return float(math.sqrt((n_e_m3 * (cls.Q_E ** 2)) / (cls.EPSILON_0 * cls.M_E)))

    @classmethod
    def debye_length_m(cls, n_e_m3: float, t_e_k: float) -> float:
        # lambda_D = sqrt(epsilon_0 * k_B * T_e / (n_e * e^2))
        return float(math.sqrt((cls.EPSILON_0 * cls.K_B * t_e_k) / (n_e_m3 * (cls.Q_E ** 2))))
