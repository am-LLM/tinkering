"""Course 241: Ramsey Interferometry & Quantum Phase Estimation Engine"""
import numpy as np

class RamseyPhaseEstimator:
    def __init__(self, t_interrogation: float = 1e-3, detuning: float = 100.0, dephasing_time_t2: float = 0.05):
        self.t = t_interrogation
        self.delta = detuning
        self.t2 = dephasing_time_t2

    def transition_probability(self, magnetic_shift: float = 0.0) -> float:
        total_detuning = self.delta + magnetic_shift
        phi = total_detuning * self.t
        decay = np.exp(-self.t / self.t2)
        # Probability of state |1> after pi/2 - free evolution - pi/2
        p1 = 0.5 * (1.0 - decay * np.cos(phi))
        return float(np.clip(p1, 0.0, 1.0))

    def estimate_phase_sensitivity(self, n_atoms: int = 1000) -> float:
        # Standard Quantum Limit (SQL) scaling
        sql = 1.0 / (np.sqrt(n_atoms) * self.t)
        return float(sql)
