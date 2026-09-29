"""Engine 39: Hill Muscle Biomechanics + High-Density Surface EMG Prosthetic."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class MuscleState:
    muscle_length_norm: float = 1.0
    activation_level: float = 0.0
    tendon_force_n: float = 0.0
    joint_angle_deg: float = 0.0

class HillMuscleProstheticEngine:
    def __init__(self, f_max_n: float = 1200.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.f_max = f_max_n
        self.state = MuscleState()

    def step_prosthetic_actuation(self, emg_signal_mv: float, dt: float = 0.02) -> Dict[str, float]:
        """Hill-type muscle model: F_m = F_max * [ a(t) * f_l(l) * f_v(v) + f_pe(l) ]."""
        # Neural activation dynamics: da/dt = (u - a) / tau
        neural_drive = np.clip(abs(emg_signal_mv) / 2.0, 0.0, 1.0)
        self.state.activation_level += (neural_drive - self.state.activation_level) * (dt / 0.05)

        # Force-length relationship: f_l = exp(-((l - 1)/0.45)^2)
        f_l = np.exp(-((self.state.muscle_length_norm - 1.0) / 0.45) ** 2)
        
        # Muscle force generation
        self.state.tendon_force_n = float(self.f_max * self.state.activation_level * f_l)
        
        # Prosthetic robotic joint torque & angle: Torque = F * lever_arm (0.04m)
        torque_nm = self.state.tendon_force_n * 0.04
        self.state.joint_angle_deg += (torque_nm / 10.0) * dt * 50.0
        self.state.joint_angle_deg = float(np.clip(self.state.joint_angle_deg, 0.0, 120.0))

        return {
            "muscle_force_n": self.state.tendon_force_n,
            "prosthetic_torque_nm": float(torque_nm),
            "joint_angle_deg": self.state.joint_angle_deg,
            "activation_level": float(self.state.activation_level)
        }
