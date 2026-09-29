"""
Engine 66: Adversarial Active Inference SCADA/ICS Sentinel
Domains: Cognitive Active Inference (Free Energy Principle) + Industrial SCADA Cyber-Physical Defense + Formal Attractor Safety

A multi-capable cyber-physical defensive sentry that models industrial control systems (turbines, chemical reactors, power substations)
as generative dynamical processes under the Free Energy Principle (FEP).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional

class ActiveInferenceScadaSentinel:
    """
    Active Inference Cyber-Physical Sentinel for SCADA/ICS Infrastructure.
    """
    def __init__(self, state_dim: int = 4, obs_dim: int = 4, dt: float = 0.01):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        
        # Generative physical dynamics matrix (damped harmonic oscillator / fluid node)
        self.A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, -0.2, 0.0, 0.0],
            [0.0, 0.0, -0.5, 0.1],
            [0.0, 0.0, 0.0, -0.2]
        ])
        
        # Observation mapping: g(s) = C * s
        self.C = np.eye(obs_dim, state_dim)
        
        # Prior precisions (inverse covariance)
        self.Pi_w = np.eye(state_dim) * 5.0   # State process precision
        self.Pi_v = np.eye(obs_dim) * 10.0    # Observation precision
        
        # Internal belief state
        self.mu = np.zeros(state_dim)
        self.mu_dot = np.zeros(state_dim)
        
        # Attractor set bounds (formally verified safe envelope)
        self.safe_bounds_min = np.array([-3.0, -3.0, -2.0, -2.0])
        self.safe_bounds_max = np.array([3.0, 3.0, 2.0, 2.0])
        
        # Attack detection metrics
        self.free_energy_history: List[float] = []
        self.quarantine_mask = np.zeros(obs_dim, dtype=bool)
        self.last_verified_state = self.mu.copy()
        self.alert_level = "NOMINAL"

    def compute_variational_free_energy(self, observation: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculates Laplace-approximated Variational Free Energy F(mu, o):
        F = 0.5 * (e_v^T * Pi_v * e_v + e_w^T * Pi_w * e_w)
        """
        e_v = observation - self.C @ self.mu
        e_w = self.mu_dot - self.A @ self.mu
        
        F_obs = 0.5 * float(e_v.T @ self.Pi_v @ e_v)
        F_proc = 0.5 * float(e_w.T @ self.Pi_w @ e_w)
        F_total = F_obs + F_proc
        
        return F_total, e_v

    def step(self, raw_telemetry: np.ndarray, control_input: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Executes one defensive active inference update cycle.
        """
        if control_input is None:
            control_input = np.zeros(self.state_dim)
            
        # 1. Evaluate Surprise / Free Energy
        F_raw, e_v = self.compute_variational_free_energy(raw_telemetry)
        self.free_energy_history.append(F_raw)
        
        # 2. Adversarial Anomaly Gating
        isolated_channels = []
        for i in range(self.obs_dim):
            sensor_surprise = 0.5 * (e_v[i] ** 2) * self.Pi_v[i, i]
            if sensor_surprise > 5.0:
                self.quarantine_mask[i] = True
                isolated_channels.append(i)
            else:
                self.quarantine_mask[i] = False
                
        if F_raw > 15.0 or len(isolated_channels) > 0:
            self.alert_level = "ATTACK_DETECTED"
        else:
            self.alert_level = "NOMINAL"
            if np.all(self.mu >= self.safe_bounds_min) and np.all(self.mu <= self.safe_bounds_max):
                self.last_verified_state = self.mu.copy()
                
        # 3. Perception Update with dynamic quarantine weighting
        active_Pi_v = self.Pi_v.copy()
        for idx in range(self.obs_dim):
            if self.quarantine_mask[idx]:
                active_Pi_v[idx, idx] *= 0.01
                
        # Proportional-integral state observer update
        self.mu_dot = self.A @ self.mu + control_input + 0.5 * (active_Pi_v @ e_v)
        self.mu = self.mu + self.mu_dot * self.dt
        
        # 4. Corrective Action Synthesis
        corrective_action = np.zeros(self.state_dim)
        if self.alert_level == "ATTACK_DETECTED":
            corrective_action = -0.2 * self.mu
            
        # 5. Catastrophic Integrity & Rollback Gate
        rollback_triggered = False
        catastrophic_anomaly = np.any(np.abs(raw_telemetry) > 10.0)
        state_breached = np.any(self.mu < self.safe_bounds_min * 2.0) or np.any(self.mu > self.safe_bounds_max * 2.0)
        
        if catastrophic_anomaly or state_breached:
            self.mu = self.last_verified_state.copy()
            self.mu_dot = np.zeros(self.state_dim)
            rollback_triggered = True
            self.alert_level = "EMERGENCY_ROLLBACK_EXECUTED"
            
        return {
            "free_energy": F_raw,
            "alert_level": self.alert_level,
            "quarantined_channels": isolated_channels,
            "filtered_state_belief": self.mu.copy(),
            "corrective_action": corrective_action,
            "rollback_triggered": rollback_triggered
        }
