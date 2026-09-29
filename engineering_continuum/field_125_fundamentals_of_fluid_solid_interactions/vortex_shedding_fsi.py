"""Course 125: Strouhal Vortex Shedding & Vortex-Induced Vibration (VIV) Engine"""
class VortexSheddingFSI:
    @staticmethod
    def shedding_frequency(flow_velocity: float, diameter: float, strouhal_number: float = 0.2) -> float:
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
        return float((strouhal_number * flow_velocity) / diameter)

    @classmethod
    def check_lock_in(cls, natural_freq: float, flow_velocity: float, diameter: float, tolerance: float = 0.1) -> bool:
        f_shed = cls.shedding_frequency(flow_velocity, diameter)
        return abs(f_shed - natural_freq) / natural_freq <= tolerance
