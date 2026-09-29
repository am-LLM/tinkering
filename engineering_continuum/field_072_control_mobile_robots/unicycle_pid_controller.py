"""Course 072: Differential Drive Unicycle Kinematic PID Path Follower"""
import math

class UnicycleController:
    def __init__(self, kp_linear=1.5, kp_angular=4.0):
        self.kp_lin = kp_linear
        self.kp_ang = kp_angular

    def compute_control(self, x: float, y: float, theta: float, target_x: float, target_y: float) -> tuple:
        dx = target_x - x
        dy = target_y - y
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        angle_err = math.atan2(math.sin(target_angle - theta), math.cos(target_angle - theta))
        v = self.kp_lin * dist * math.cos(angle_err)
        w = self.kp_ang * angle_err
        return float(v), float(w)

    @staticmethod
    def step_kinematics(x, y, theta, v, w, dt=0.1):
        x_next = x + v * math.cos(theta) * dt
        y_next = y + v * math.sin(theta) * dt
        theta_next = theta + w * dt
        return float(x_next), float(y_next), float(theta_next)
