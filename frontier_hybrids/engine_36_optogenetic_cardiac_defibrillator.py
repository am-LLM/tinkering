"""Engine 36: Optogenetic Cardiac Reset + Aliev-Panfilov Spiral Wave Defibrillation."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class OptogeneticCardiacEngine:
    def __init__(self, size: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.size = size
        # Action potential transmembrane voltage u, recovery variable v
        self.u = self.rng.uniform(0.0, 0.8, size=(size, size))
        self.v = self.rng.uniform(0.0, 0.4, size=(size, size))
        self.k = 8.0
        self.a = 0.15

    def step_cardiac_optogenetics(self, blue_light_flux_mw_mm2: float = 0.0, dt: float = 0.02) -> Dict[str, float]:
        """Aliev-Panfilov model with Channelrhodopsin-2 (ChR2) optogenetic photocurrent."""
        laplacian_u = (
            np.roll(self.u, 1, axis=0) + np.roll(self.u, -1, axis=0) +
            np.roll(self.u, 1, axis=1) + np.roll(self.u, -1, axis=1) - 4 * self.u
        )
        # ChR2 depolarizing inward current
        i_chr2 = blue_light_flux_mw_mm2 * 0.3
        
        # du/dt = -k*u*(u-a)*(u-1) - u*v + I_opt + D*laplacian(u)
        du = (-self.k * self.u * (self.u - self.a) * (self.u - 1.0) - self.u * self.v + i_chr2 + 0.1 * laplacian_u) * dt
        # dv/dt = (epsilon + mu1*v/(mu2 + u)) * (-v - k*u*(u - b - 1))
        dv = (0.01 + 0.1 * self.v / (0.5 + self.u + 1e-6)) * (-self.v - self.k * self.u * (self.u - 1.1)) * dt

        self.u = np.clip(self.u + du, 0.0, 1.2)
        self.v = np.clip(self.v + dv, 0.0, 1.0)

        # Fibrillation rotor entropy (variance of excitation)
        fibrillation_entropy = float(np.var(self.u))
        return {
            "mean_action_potential_u": float(np.mean(self.u)),
            "fibrillation_entropy": fibrillation_entropy,
            "optogenetic_current": float(i_chr2)
        }
