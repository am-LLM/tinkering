"""Course 286: Hill-Type Musculoskeletal Muscle-Tendon Gait Dynamics Solver"""
import numpy as np

class HillMuscleModel:
    def __init__(self, f_max: float = 1500.0, l_slack: float = 0.25, l_opt: float = 0.08, v_max: float = 10.0):
        self.f_max = f_max
        self.l_slack = l_slack
        self.l_opt = l_opt
        self.v_max = v_max

    def force_length_active(self, l_ce: float) -> float:
        # Normalized Gaussian active force-length curve
        l_norm = l_ce / self.l_opt
        fl = np.exp(-((l_norm - 1.0) / 0.45)**2)
        return float(fl)

    def tendon_force(self, l_tendon: float) -> float:
        strain = (l_tendon - self.l_slack) / self.l_slack
        if strain <= 0:
            return 0.0
        # Quadratic toe region up to 4% strain, then linear
        if strain < 0.04:
            f_t_norm = (strain / 0.04)**2
        else:
            f_t_norm = (strain / 0.04)
        return float(f_t_norm * self.f_max)

    def compute_muscle_force(self, activation: float, l_ce: float, l_tendon: float) -> float:
        act = np.clip(activation, 0.0, 1.0)
        f_active = act * self.force_length_active(l_ce) * self.f_max
        f_t = self.tendon_force(l_tendon)
        return float(min(f_active, f_t) if f_t > 0 else f_active)
