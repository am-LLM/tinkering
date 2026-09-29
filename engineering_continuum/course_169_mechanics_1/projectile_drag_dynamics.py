"""Course 169: 2D Projectile Motion Integrator with Quadratic Aerodynamic Drag"""
import math

class ProjectileDragDynamics:
    def __init__(self, mass_kg: float = 1.0, drag_coeff: float = 0.05, g: float = 9.81):
        self.m, self.k, self.g = mass_kg, drag_coeff, g

    def step(self, x: float, y: float, vx: float, vy: float, dt: float = 0.01) -> tuple:
        v = math.hypot(vx, vy)
        f_drag_x = -self.k * v * vx
        f_drag_y = -self.m * self.g - self.k * v * vy
        
        ax = f_drag_x / self.m
        ay = f_drag_y / self.m
        
        x_next = x + vx * dt
        y_next = y + vy * dt
        vx_next = vx + ax * dt
        vy_next = vy + ay * dt
        return x_next, y_next, vx_next, vy_next
