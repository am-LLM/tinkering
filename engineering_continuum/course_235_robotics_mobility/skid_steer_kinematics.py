"""Course 235: 4-Wheel Skid-Steer Mobile Robot Kinematic Differential Drive Model"""
class SkidSteerKinematics:
    def __init__(self, track_width_m: float = 0.6, wheel_radius_m: float = 0.1):
        self.b = track_width_m
        self.r = wheel_radius_m

    def forward_kinematics(self, omega_left: float, omega_right: float) -> tuple:
        v_l = omega_left * self.r
        v_r = omega_right * self.r
        v_lin = (v_r + v_l) / 2.0
        omega_ang = (v_r - v_l) / self.b
        return float(v_lin), float(omega_ang)
