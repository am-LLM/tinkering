"""
Frontier Quant Hybrid Engine 1: Astrodynamic 3-Body Liquidity Potential & Jacobi Invariant Sentry (A-LOB)
Reconstructs the true physical topology of limit order books using restricted 3-body celestial mechanics,
calculating Lagrange equilibrium points (L1-L5), Hill spheres, and Zero-Velocity Surface barriers.
"""
import numpy as np
from typing import Dict, Tuple, List

class AstrodynamicLOB:
    def __init__(self, g_const: float = 1.0, omega_synodic: float = 1.0):
        self.g = g_const
        self.omega = omega_synodic

    def compute_lagrange_points(self, m1: float, m2: float, d: float) -> Dict[str, Tuple[float, float]]:
        """Computes collinear (L1, L2, L3) and triangular (L4, L5) Lagrange equilibrium points."""
        mu = m2 / (m1 + m2)
        # Position of primary M1 at (-mu*d, 0) and secondary M2 at ((1-mu)*d, 0)
        x_m1 = -mu * d
        x_m2 = (1.0 - mu) * d
        
        # Approximate L1 (between M1 and M2)
        r_hill = d * (mu / 3.0) ** (1.0 / 3.0)
        l1_x = x_m2 - r_hill
        l2_x = x_m2 + r_hill
        l3_x = x_m1 - d * (1.0 + 5.0 * mu / 12.0)
        
        # Triangular points L4 and L5 form equilateral triangles
        l4_x = (x_m1 + x_m2) / 2.0
        l4_y = d * np.sqrt(3) / 2.0
        l5_x = l4_x
        l5_y = -l4_y
        
        return {
            "L1": (float(l1_x), 0.0),
            "L2": (float(l2_x), 0.0),
            "L3": (float(l3_x), 0.0),
            "L4": (float(l4_x), float(l4_y)),
            "L5": (float(l5_x), float(l5_y)),
            "Hill_Radius": float(r_hill)
        }

    def effective_potential(self, x: float, y: float, m1: float, m2: float, d: float) -> float:
        """Effective potential Omega(x,y) in the rotating frame."""
        mu = m2 / (m1 + m2)
        x_m1 = -mu * d
        x_m2 = (1.0 - mu) * d
        
        r1 = max(np.sqrt((x - x_m1)**2 + y**2), 1e-4)
        r2 = max(np.sqrt((x - x_m2)**2 + y**2), 1e-4)
        
        centrifugal = 0.5 * (self.omega ** 2) * (x**2 + y**2)
        gravitational = self.g * (m1 / r1 + m2 / r2)
        return float(centrifugal + gravitational)

    def jacobi_constant(self, x: float, y: float, vx: float, vy: float, m1: float, m2: float, d: float) -> float:
        """Jacobi integral of motion: C = 2*Omega(x,y) - (vx^2 + vy^2)."""
        omega_val = self.effective_potential(x, y, m1, m2, d)
        v_squared = vx**2 + vy**2
        return float(2.0 * omega_val - v_squared)

    def is_region_accessible(self, test_x: float, test_y: float, current_c: float, m1: float, m2: float, d: float) -> bool:
        """Determines if a price-coordinate state is physically accessible under zero-velocity curves."""
        omega_val = self.effective_potential(test_x, test_y, m1, m2, d)
        # v^2 = 2*Omega - C >= 0 for real physical motion
        v_sq = 2.0 * omega_val - current_c
        return v_sq >= 0.0
