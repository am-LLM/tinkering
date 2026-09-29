"""Course 233: Quadrotor 6-DOF Nonlinear Rigid Body Flight Dynamics Integrator"""
import numpy as np

class Quadrotor6DOFFlight:
    def __init__(self, mass: float = 1.0, g: float = 9.81):
        self.m, self.g = mass, g

    def step_linear_dynamics(self, pos: np.ndarray, vel: np.ndarray, thrust_total: float, roll: float, pitch: float, dt: float = 0.01) -> tuple:
        ax = thrust_total * (np.sin(pitch)) / self.m
        ay = -thrust_total * (np.sin(roll)) / self.m
        az = (thrust_total * (np.cos(roll) * np.cos(pitch)) - self.m * self.g) / self.m
        
        acc = np.array([ax, ay, az])
        vel_next = vel + acc * dt
        pos_next = pos + vel_next * dt
        return pos_next, vel_next
