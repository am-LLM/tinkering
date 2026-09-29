"""Engine 06: Continuous RL Actor-Critic + Hypersonic MHD Plasma Control."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple

@dataclass
class HypersonicState:
    mach: float = 7.0
    altitude_km: float = 35.0
    stagnation_temp_k: float = 2400.0
    magnetic_b_field_tesla: float = 0.0
    plasma_conductivity_s_m: float = 120.0
    heat_flux_mw_m2: float = 2.5

class HypersonicRLMHDControlEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = HypersonicState()
        # Actor weights: state -> B-field actuation
        self.actor_weights = self.rng.normal(0.0, 0.1, size=(4,))
        # Critic weights: state -> Value
        self.critic_weights = self.rng.normal(0.0, 0.1, size=(4,))

    def get_features(self) -> np.ndarray:
        return np.array([
            self.state.mach / 10.0,
            self.state.altitude_km / 50.0,
            self.state.stagnation_temp_k / 3000.0,
            self.state.heat_flux_mw_m2 / 5.0
        ])

    def step(self, dt: float = 0.05) -> Dict[str, float]:
        features = self.get_features()
        # Actor action: magnetic field excitation (0 to 3 Tesla)
        raw_action = float(np.dot(self.actor_weights, features))
        b_action = float(np.clip(1.5 * (np.tanh(raw_action) + 1.0), 0.0, 3.0))
        self.state.magnetic_b_field_tesla = b_action

        # Hypersonic Aerothermodynamics + MHD Lorentz deceleration
        # Interaction parameter: Q = sigma * B^2 * L / (rho * u)
        lorentz_cooling_factor = 1.0 / (1.0 + 0.3 * (b_action ** 2))
        self.state.heat_flux_mw_m2 = float(2.5 * lorentz_cooling_factor + self.rng.normal(0, 0.05))
        self.state.stagnation_temp_k = float(2400.0 * lorentz_cooling_factor + 10.0 * self.state.mach)

        # Reward: minimize heat flux and high temperature while conserving magnetic power
        reward = -(self.state.heat_flux_mw_m2 + 0.1 * (b_action ** 2))

        # Critic evaluation & TD error
        v_curr = float(np.dot(self.critic_weights, features))
        features_next = self.get_features()
        v_next = float(np.dot(self.critic_weights, features_next))
        td_error = reward + 0.95 * v_next - v_curr

        # Gradient updates
        self.actor_weights += 0.01 * td_error * features
        self.critic_weights += 0.02 * td_error * features

        return {
            "mach": self.state.mach,
            "b_field_tesla": b_action,
            "heat_flux_mw_m2": self.state.heat_flux_mw_m2,
            "stagnation_temp_k": self.state.stagnation_temp_k,
            "td_error": float(td_error)
        }
