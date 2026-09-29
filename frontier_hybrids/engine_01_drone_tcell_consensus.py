"""Engine 01: GPS-Denied Drone Swarm Consensus + T-Cell Immune Networks."""
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List

@dataclass
class DroneState:
    drone_id: int
    position: np.ndarray
    velocity: np.ndarray
    affinity: float = 1.0
    is_byzantine: bool = False
    is_suppressed: bool = False
    epitope_signature: np.ndarray = field(default_factory=lambda: np.zeros(4))

@dataclass
class ImmuneReceptor:
    receptor_id: int
    target_pattern: np.ndarray
    affinity_threshold: float = 0.6
    somatic_mutation_rate: float = 0.05

class DroneTCellConsensusEngine:
    def __init__(self, num_drones: int = 10, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_drones = num_drones
        self.drones: Dict[int, DroneState] = {}
        self.receptors: List[ImmuneReceptor] = []
        self._init_swarm()
        self._init_tcell_pool()

    def _init_swarm(self):
        for i in range(self.num_drones):
            pos = self.rng.uniform(-10.0, 10.0, size=3)
            vel = self.rng.uniform(-0.5, 0.5, size=3)
            # Shared swarm signature pattern with small variance
            sig = np.array([1.0, 0.5, -0.5, 0.2]) + self.rng.normal(0.0, 0.05, size=4)
            sig /= (np.linalg.norm(sig) + 1e-9)
            self.drones[i] = DroneState(drone_id=i, position=pos, velocity=vel, epitope_signature=sig)

    def _init_tcell_pool(self, num_receptors: int = 8):
        for r_id in range(num_receptors):
            pat = np.array([1.0, 0.5, -0.5, 0.2]) + self.rng.normal(0.0, 0.1, size=4)
            pat /= (np.linalg.norm(pat) + 1e-9)
            self.receptors.append(ImmuneReceptor(receptor_id=r_id, target_pattern=pat))

    def inject_byzantine_fault(self, drone_id: int, malicious_velocity: np.ndarray):
        if drone_id in self.drones:
            self.drones[drone_id].is_byzantine = True
            self.drones[drone_id].velocity = np.array(malicious_velocity, dtype=float)
            # Severe epitope anomaly for byzantine
            corrupt_sig = np.array([-1.0, -1.0, 1.0, -1.0])
            self.drones[drone_id].epitope_signature = corrupt_sig / (np.linalg.norm(corrupt_sig) + 1e-9)

    def step_consensus(self, dt: float = 0.1) -> Dict[str, float]:
        unsuppressed = [d for d in self.drones.values() if not d.is_suppressed]
        if unsuppressed:
            mean_sig = np.mean([d.epitope_signature for d in unsuppressed], axis=0)
            mean_sig /= (np.linalg.norm(mean_sig) + 1e-9)
        else:
            mean_sig = np.array([1.0, 0.0, 0.0, 0.0])

        for d in self.drones.values():
            deviation = float(np.linalg.norm(d.epitope_signature - mean_sig))
            affinities = [float(np.dot(rec.target_pattern, d.epitope_signature)) for rec in self.receptors]
            best_aff = max(affinities) if affinities else 0.0
            d.affinity = best_aff
            if deviation > 0.8 or d.affinity < 0.2:
                d.is_suppressed = True
            else:
                d.is_suppressed = False

        active = [d for d in self.drones.values() if not d.is_suppressed]
        if not active:
            return {"active_drones": 0, "velocity_variance": 0.0}

        avg_vel = np.mean([d.velocity for d in active], axis=0)
        avg_pos = np.mean([d.position for d in active], axis=0)

        for d in active:
            if not d.is_byzantine:
                cohesion = (avg_pos - d.position) * 0.05
                alignment = (avg_vel - d.velocity) * 0.2
                d.velocity += (cohesion + alignment) * dt
                speed = np.linalg.norm(d.velocity)
                if speed > 2.0:
                    d.velocity = (d.velocity / speed) * 2.0
            d.position += d.velocity * dt

        active_vels = [d.velocity for d in active]
        vel_var = float(np.var(active_vels)) if len(active_vels) > 1 else 0.0
        return {"active_drones": len(active), "velocity_variance": vel_var}
