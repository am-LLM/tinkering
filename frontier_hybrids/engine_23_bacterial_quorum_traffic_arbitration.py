"""Engine 23: Bacterial Quorum Sensing + Intersection Traffic Arbitration."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class VehicleAgent:
    vehicle_id: int
    approach_lane: str # "NORTH", "SOUTH", "EAST", "WEST"
    autoinducer_ai2_conc: float = 0.0
    cleared_intersection: bool = False

class BacterialQuorumTrafficEngine:
    def __init__(self, num_vehicles: int = 12, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        lanes = ["NORTH", "SOUTH", "EAST", "WEST"]
        self.vehicles = [
            VehicleAgent(vehicle_id=i, approach_lane=lanes[i % 4],
                         autoinducer_ai2_conc=float(self.rng.uniform(0.1, 0.5)))
            for i in range(num_vehicles)
        ]
        self.quorum_threshold = 2.5 # Critical threshold to trigger green light gene expression

    def step_arbitration(self, dt: float = 0.1) -> Dict[str, float]:
        """Simulate AI-2 autoinducer accumulation per directional cluster."""
        lane_clusters: Dict[str, float] = {"NORTH": 0.0, "SOUTH": 0.0, "EAST": 0.0, "WEST": 0.0}
        for v in self.vehicles:
            if not v.cleared_intersection:
                v.autoinducer_ai2_conc += 0.2 * dt # Continual secretion while waiting
                lane_clusters[v.approach_lane] += v.autoinducer_ai2_conc

        # Determine winning quorum (NS axis vs EW axis)
        ns_quorum = lane_clusters["NORTH"] + lane_clusters["SOUTH"]
        ew_quorum = lane_clusters["EAST"] + lane_clusters["WEST"]

        granted_axis = "NS" if ns_quorum >= ew_quorum else "EW"
        cleared_count = 0
        for v in self.vehicles:
            if not v.cleared_intersection:
                if (granted_axis == "NS" and v.approach_lane in ["NORTH", "SOUTH"]) or                    (granted_axis == "EW" and v.approach_lane in ["EAST", "WEST"]):
                    if max(ns_quorum, ew_quorum) >= self.quorum_threshold:
                        v.cleared_intersection = True
                        cleared_count += 1

        active_remaining = sum(1 for v in self.vehicles if not v.cleared_intersection)
        return {
            "ns_quorum_conc": float(ns_quorum),
            "ew_quorum_conc": float(ew_quorum),
            "remaining_vehicles": float(active_remaining),
            "cleared_vehicles": float(len(self.vehicles) - active_remaining)
        }
