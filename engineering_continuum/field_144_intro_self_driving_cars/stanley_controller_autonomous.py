"""Course 144: Stanley Front-Axle Lateral Path Tracking Steering Controller"""
import math

class StanleyController:
    def __init__(self, k_gain: float = 0.5, softening_const: float = 1.0):
        self.k = k_gain
        self.k_soft = softening_const

    def compute_steering_angle(self, heading_err_rad: float, cross_track_err: float, velocity_mps: float) -> float:
        # delta = psi + arctan(k * e / (v + k_soft))
        cross_track_term = math.atan2(self.k * cross_track_err, velocity_mps + self.k_soft)
        delta = heading_err_rad + cross_track_term
        return float(max(-math.pi/4, min(math.pi/4, delta)))
