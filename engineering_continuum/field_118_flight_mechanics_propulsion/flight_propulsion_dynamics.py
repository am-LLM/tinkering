"""Course 118: Tsiolkovsky Rocket Equation & Jet Propulsion Specific Impulse"""
import math

class FlightPropulsionDynamics:
    @staticmethod
    def tsiolkovsky_delta_v(isp_sec: float, m_initial: float, m_dry: float, g0: float = 9.80665) -> float:
        if m_dry <= 0 or m_initial < m_dry:
            raise ValueError("Invalid mass ratio")
        v_exhaust = isp_sec * g0
        return float(v_exhaust * math.log(m_initial / m_dry))

    @staticmethod
    def brayton_thermal_efficiency(pressure_ratio: float, gamma: float = 1.4) -> float:
        return float(1.0 - (1.0 / (pressure_ratio ** ((gamma - 1.0) / gamma))))
