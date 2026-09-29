"""Engine 30: Kramers Stochastic Resonance + MEMS Sensor Sub-Threshold Amplifier."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class StochasticResonanceMEMSEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Bistable potential V(x) = -a/2 * x^2 + b/4 * x^4
        self.a = 1.0
        self.b = 1.0
        self.state_x = 0.0

    def step_resonance(self, sub_threshold_signal: float, noise_intensity_d: float = 0.35, dt: float = 0.01) -> Dict[str, float]:
        """Langevin equation: dx/dt = a*x - b*x^3 + A*cos(omega*t) + sqrt(2*D)*xi(t)."""
        drift = self.a * self.state_x - self.b * (self.state_x ** 3) + sub_threshold_signal
        noise = np.sqrt(2 * noise_intensity_d) * self.rng.normal(0, 1) / np.sqrt(dt)
        self.state_x += (drift + noise) * dt

        # Kramers barrier height Delta V = a^2 / (4*b) = 0.25
        kramers_rate = (self.a / (np.sqrt(2) * np.pi)) * np.exp(-0.25 / (noise_intensity_d + 1e-6))

        return {
            "output_state_x": float(self.state_x),
            "kramers_escape_rate": float(kramers_rate),
            "snr_amplification_db": float(10.0 * np.log10(1.0 + abs(self.state_x) / (abs(sub_threshold_signal) + 1e-6)))
        }
