"""
Apollo TMR & FDIR: Triple Modular Redundant (TMR) Flight Executive with Dynamic Sensor Scavenging & Brownout Self-Healing
-------------------------------------------------------------------------------------------------------------------------
Implements NASA-grade fault-tolerant avionics control:
1. 2-out-of-3 Majority Voting with dynamic drift detection and configurable tolerance bands.
2. Dynamic Sensor Scavenging across cross-strapped avionics data buses to synthesize healthy sensor frames from fragmented redundant hardware.
3. Brownout & Single-Event Upset (SEU) Self-Healing with automatic checkpointing, state scrubbing, and hot-quorum re-synchronization.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any


class NodeHealth(Enum):
    HEALTHY = auto()
    TRANSIENT_ANOMALY = auto()
    DEGRADED = auto()
    HARD_FAULT = auto()
    BROWNOUT = auto()


class SystemMode(Enum):
    TRIPLEX = auto()    # Full 3-node TMR consensus
    DUPLEX = auto()     # 2-node consensus (1 node disqualified)
    SIMPLEX = auto()    # 1-node fail-safe operation (2 nodes disqualified)
    SAFE_HOLD = auto()  # All channels divergent or unrecoverable, emergency attitude hold


@dataclass
class SensorPacket:
    sensor_id: str
    timestamp: float
    data: np.ndarray
    healthy: bool = True
    checksum_valid: bool = True


@dataclass
class FlightStateSnapshot:
    step: int
    timestamp: float
    attitude_quat: np.ndarray
    angular_rate: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    actuator_cmd: np.ndarray


class FlightComputerNode:
    """
    Simulated Flight Computer Node (A, B, or C) executing control laws.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.health = NodeHealth.HEALTHY
        self.consecutive_mismatches: int = 0
        self.sensor_inputs: Dict[str, SensorPacket] = {}
        
        # Internal state vector [p(3), v(3), q(4), w(3)]
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.attitude_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.angular_rate = np.zeros(3, dtype=np.float64)
        
        # Last calculated control command (e.g. 4-thruster PWM duty cycles [0..1])
        self.control_command = np.zeros(4, dtype=np.float64)
        self.checkpoint_state: Optional[FlightStateSnapshot] = None

    def execute_control_cycle(self, step: int, dt: float) -> np.ndarray:
        """
        Executes internal guidance, navigation & control (GNC) loop.
        """
        if self.health in [NodeHealth.HARD_FAULT, NodeHealth.BROWNOUT]:
            # Degraded / corrupted output
            return self.control_command * 0.0 + np.random.normal(999.0, 10.0, size=4)
            
        # Basic control law based on available sensors
        imu = self.sensor_inputs.get("IMU")
        if imu and imu.healthy:
            acc = imu.data[0:3]
            gyro = imu.data[3:6]
            self.angular_rate = gyro.copy()
            self.velocity += acc * dt
            self.position += self.velocity * dt
            
        # Calculate nominal PD control effort
        target_rate = np.array([0.0, 0.0, 0.0])
        rate_error = target_rate - self.angular_rate
        kp = 0.5
        effort = np.clip(0.5 + kp * np.sum(rate_error), 0.0, 1.0)
        self.control_command = np.array([effort, effort, effort, effort], dtype=np.float64)
        return self.control_command.copy()

    def create_checkpoint(self, step: int, timestamp: float):
        """Creates atomic memory snapshot for SEU recovery."""
        self.checkpoint_state = FlightStateSnapshot(
            step=step,
            timestamp=timestamp,
            attitude_quat=self.attitude_quat.copy(),
            angular_rate=self.angular_rate.copy(),
            velocity=self.velocity.copy(),
            position=self.position.copy(),
            actuator_cmd=self.control_command.copy()
        )

    def restore_from_state(self, snapshot: FlightStateSnapshot):
        """Hot-sync restores state vector from verified quorum snapshot."""
        self.attitude_quat = snapshot.attitude_quat.copy()
        self.angular_rate = snapshot.angular_rate.copy()
        self.velocity = snapshot.velocity.copy()
        self.position = snapshot.position.copy()
        self.control_command = snapshot.actuator_cmd.copy()
        self.health = NodeHealth.HEALTHY
        self.consecutive_mismatches = 0


class DynamicSensorScavenger:
    """
    Cross-strapped sensor bus aggregator that scavenges healthy redundant sensor
    channels across fragmented nodes to maintain complete sensor frames.
    """
    def __init__(self):
        self.scavenged_sensor_pool: Dict[str, SensorPacket] = {}

    def ingest_node_sensors(self, all_nodes: List[FlightComputerNode]):
        """Collects and validates all sensor packets across nodes."""
        self.scavenged_sensor_pool.clear()
        for node in all_nodes:
            for sensor_type, packet in node.sensor_inputs.items():
                if packet.healthy and packet.checksum_valid:
                    # If not yet registered or higher priority, register in global scavenged pool
                    if sensor_type not in self.scavenged_sensor_pool:
                        self.scavenged_sensor_pool[sensor_type] = packet

    def distribute_scavenged_sensors(self, all_nodes: List[FlightComputerNode]):
        """Provides missing or failed sensor channels to any degraded nodes from the pool."""
        for node in all_nodes:
            for sensor_type, scavenged_packet in self.scavenged_sensor_pool.items():
                current_packet = node.sensor_inputs.get(sensor_type)
                if current_packet is None or not current_packet.healthy or not current_packet.checksum_valid:
                    # Dynamically scavenge and replace with verified packet
                    node.sensor_inputs[sensor_type] = SensorPacket(
                        sensor_id=f"Scavenged_{scavenged_packet.sensor_id}",
                        timestamp=scavenged_packet.timestamp,
                        data=scavenged_packet.data.copy(),
                        healthy=True,
                        checksum_valid=True
                    )


class ApolloTMRFlightExecutive:
    """
    Triple Modular Redundant (TMR) 2-out-of-3 majority voting flight executive with FDIR.
    """
    def __init__(self, tolerance: float = 0.05, max_transient_mismatches: int = 3):
        self.nodes = [
            FlightComputerNode("NodeA"),
            FlightComputerNode("NodeB"),
            FlightComputerNode("NodeC")
        ]
        self.tolerance = float(tolerance)
        self.max_transient_mismatches = int(max_transient_mismatches)
        self.scavenger = DynamicSensorScavenger()
        self.current_mode = SystemMode.TRIPLEX
        self.step_count = 0
        self.fdir_events_log: List[Dict[str, Any]] = []

    def inject_seu_bitflip(self, node_idx: int):
        """Simulates cosmic ray Single-Event Upset / memory corruption in specified node."""
        node = self.nodes[node_idx]
        node.angular_rate += np.array([100.0, -250.0, 500.0])
        node.control_command += np.array([5.0, -5.0, 10.0, -10.0])
        node.health = NodeHealth.TRANSIENT_ANOMALY

    def inject_brownout(self, node_idx: int):
        """Simulates sudden power drop / brownout on specified node."""
        node = self.nodes[node_idx]
        node.health = NodeHealth.BROWNOUT
        node.control_command.fill(np.nan)

    def step(self, dt: float = 0.01) -> Tuple[np.ndarray, SystemMode]:
        """
        Executes one full TMR synchronized control cycle:
        1. Sensor Scavenging
        2. GNC execution on all active nodes
        3. 2-out-of-3 Majority Voting
        4. FDIR Isolation & Brownout Self-Healing
        """
        self.step_count += 1
        
        # 1. Dynamic Sensor Scavenging
        self.scavenger.ingest_node_sensors(self.nodes)
        self.scavenger.distribute_scavenged_sensors(self.nodes)
        
        # 2. Parallel GNC execution
        cmds = []
        for node in self.nodes:
            cmd = node.execute_control_cycle(self.step_count, dt)
            cmds.append(cmd)
            
        # 3. 2-out-of-3 Majority Voting & Consensus
        voted_cmd, agreeing_nodes, diverging_nodes = self._vote_majority(cmds)
        
        # 4. FDIR State Machine and Self-Healing
        self._fdir_reconcile(voted_cmd, agreeing_nodes, diverging_nodes)
        
        # 5. Checkpointing on healthy agreeing nodes
        consensus_snapshot = FlightStateSnapshot(
            step=self.step_count,
            timestamp=self.step_count * dt,
            attitude_quat=self.nodes[agreeing_nodes[0]].attitude_quat.copy() if agreeing_nodes else np.array([1.,0.,0.,0.]),
            angular_rate=self.nodes[agreeing_nodes[0]].angular_rate.copy() if agreeing_nodes else np.zeros(3),
            velocity=self.nodes[agreeing_nodes[0]].velocity.copy() if agreeing_nodes else np.zeros(3),
            position=self.nodes[agreeing_nodes[0]].position.copy() if agreeing_nodes else np.zeros(3),
            actuator_cmd=voted_cmd.copy()
        )
        for idx in agreeing_nodes:
            self.nodes[idx].create_checkpoint(self.step_count, self.step_count * dt)
            
        # 6. Brownout / SEU Self-Healing: Re-sync diverging/corrupted nodes from consensus
        for idx in diverging_nodes:
            if self.nodes[idx].health in [NodeHealth.TRANSIENT_ANOMALY, NodeHealth.BROWNOUT]:
                # Perform hot-restart & state reconstitution from quorum
                self.nodes[idx].restore_from_state(consensus_snapshot)
                self.fdir_events_log.append({
                    "step": self.step_count,
                    "event": "SELF_HEAL_RESYNC",
                    "target_node": self.nodes[idx].node_id
                })
                
        return voted_cmd, self.current_mode

    def _vote_majority(self, cmds: List[np.ndarray]) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Calculates consensus among the 3 channels.
        Returns: (consensus_cmd, agreeing_node_indices, diverging_node_indices)
        """
        active_indices = [i for i, n in enumerate(self.nodes) if n.health != NodeHealth.HARD_FAULT]
        
        if len(active_indices) >= 3:
            # Full Triplex voting: test all pairs (0,1), (0,2), (1,2)
            c0, c1, c2 = cmds[0], cmds[1], cmds[2]
            d01 = np.max(np.abs(c0 - c1)) if not (np.any(np.isnan(c0)) or np.any(np.isnan(c1))) else 999.0
            d02 = np.max(np.abs(c0 - c2)) if not (np.any(np.isnan(c0)) or np.any(np.isnan(c2))) else 999.0
            d12 = np.max(np.abs(c1 - c2)) if not (np.any(np.isnan(c1)) or np.any(np.isnan(c2))) else 999.0
            
            # Case 1: All 3 agree
            if d01 <= self.tolerance and d02 <= self.tolerance and d12 <= self.tolerance:
                self.current_mode = SystemMode.TRIPLEX
                return (c0 + c1 + c2) / 3.0, [0, 1, 2], []
                
            # Case 2: 0 & 1 agree (2 is outlier)
            if d01 <= self.tolerance:
                self.current_mode = SystemMode.TRIPLEX
                return (c0 + c1) / 2.0, [0, 1], [2]
                
            # Case 3: 0 & 2 agree (1 is outlier)
            if d02 <= self.tolerance:
                self.current_mode = SystemMode.TRIPLEX
                return (c0 + c2) / 2.0, [0, 2], [1]
                
            # Case 4: 1 & 2 agree (0 is outlier)
            if d12 <= self.tolerance:
                self.current_mode = SystemMode.TRIPLEX
                return (c1 + c2) / 2.0, [1, 2], [0]
                
            # Case 5: All 3 mutually disagree
            self.current_mode = SystemMode.SAFE_HOLD
            return np.zeros(4, dtype=np.float64), [], [0, 1, 2]
            
        elif len(active_indices) == 2:
            i, j = active_indices[0], active_indices[1]
            ci, cj = cmds[i], cmds[j]
            dij = np.max(np.abs(ci - cj)) if not (np.any(np.isnan(ci)) or np.any(np.isnan(cj))) else 999.0
            if dij <= self.tolerance:
                self.current_mode = SystemMode.DUPLEX
                return (ci + cj) / 2.0, [i, j], [k for k in range(3) if k not in [i, j]]
            else:
                self.current_mode = SystemMode.SAFE_HOLD
                return np.zeros(4, dtype=np.float64), [], [i, j]
                
        elif len(active_indices) == 1:
            idx = active_indices[0]
            self.current_mode = SystemMode.SIMPLEX
            return cmds[idx], [idx], [k for k in range(3) if k != idx]
            
        else:
            self.current_mode = SystemMode.SAFE_HOLD
            return np.zeros(4, dtype=np.float64), [], [0, 1, 2]

    def _fdir_reconcile(self, voted_cmd: np.ndarray, agreeing_nodes: List[int], diverging_nodes: List[int]):
        """Updates error counters and isolates permanently faulty nodes."""
        for idx in agreeing_nodes:
            self.nodes[idx].consecutive_mismatches = max(0, self.nodes[idx].consecutive_mismatches - 1)
            if self.nodes[idx].health == NodeHealth.TRANSIENT_ANOMALY:
                self.nodes[idx].health = NodeHealth.HEALTHY
                
        for idx in diverging_nodes:
            self.nodes[idx].consecutive_mismatches += 1
            if self.nodes[idx].consecutive_mismatches >= self.max_transient_mismatches:
                self.nodes[idx].health = NodeHealth.HARD_FAULT
                self.fdir_events_log.append({
                    "step": self.step_count,
                    "event": "HARD_FAULT_ISOLATION",
                    "target_node": self.nodes[idx].node_id
                })
            else:
                if self.nodes[idx].health != NodeHealth.BROWNOUT:
                    self.nodes[idx].health = NodeHealth.TRANSIENT_ANOMALY
