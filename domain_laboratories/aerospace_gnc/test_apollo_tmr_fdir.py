"""
Unit Tests for Apollo TMR FDIR: Triple Modular Redundant Executive with Dynamic Sensor Scavenging & Self-Healing
"""

import pytest
import numpy as np
from apollo_tmr_fdir import (
    ApolloTMRFlightExecutive, NodeHealth, SystemMode,
    SensorPacket, FlightComputerNode, DynamicSensorScavenger, FlightStateSnapshot
)


def test_apollo_tmr_nominal_consensus():
    tmr = ApolloTMRFlightExecutive()
    for node in tmr.nodes:
        node.sensor_inputs["IMU"] = SensorPacket(
            sensor_id="IMU_1",
            timestamp=0.0,
            data=np.array([0.0, 0.0, 9.80665, 0.0, 0.0, 0.0]),
            healthy=True
        )
        
    cmd, mode = tmr.step()
    assert mode == SystemMode.TRIPLEX
    assert len(cmd) == 4
    assert np.all(cmd >= 0.0) and np.all(cmd <= 1.0)
    for node in tmr.nodes:
        assert node.health == NodeHealth.HEALTHY


def test_apollo_tmr_seu_fault_isolation():
    tmr = ApolloTMRFlightExecutive()
    for node in tmr.nodes:
        node.sensor_inputs["IMU"] = SensorPacket(
            sensor_id="IMU_1",
            timestamp=0.0,
            data=np.array([0.0, 0.0, 9.80665, 0.0, 0.0, 0.0]),
            healthy=True
        )
        
    tmr.step()
    tmr.inject_seu_bitflip(1)
    assert tmr.nodes[1].health == NodeHealth.TRANSIENT_ANOMALY
    
    voted_cmd, mode = tmr.step()
    assert mode == SystemMode.TRIPLEX
    assert tmr.nodes[1].health == NodeHealth.HEALTHY


def test_apollo_tmr_brownout_self_healing():
    tmr = ApolloTMRFlightExecutive()
    for node in tmr.nodes:
        node.sensor_inputs["IMU"] = SensorPacket(
            sensor_id="IMU_1",
            timestamp=0.0,
            data=np.array([0.0, 0.0, 9.80665, 0.0, 0.0, 0.0]),
            healthy=True
        )
        
    tmr.step()
    tmr.inject_brownout(2)
    assert tmr.nodes[2].health == NodeHealth.BROWNOUT
    
    voted_cmd, mode = tmr.step()
    assert len(tmr.fdir_events_log) > 0
    assert tmr.nodes[2].health == NodeHealth.HEALTHY


def test_dynamic_sensor_scavenging():
    tmr = ApolloTMRFlightExecutive()
    
    tmr.nodes[0].sensor_inputs["IMU"] = SensorPacket("IMU_A", 0.0, np.array([0., 0., 9.8, 0., 0., 0.]), healthy=True)
    tmr.nodes[0].sensor_inputs["BARO"] = SensorPacket("BARO_A", 0.0, np.array([0.0]), healthy=False)
    
    tmr.nodes[1].sensor_inputs["IMU"] = SensorPacket("IMU_B", 0.0, np.array([0., 0., 0., 0., 0., 0.]), healthy=False)
    tmr.nodes[1].sensor_inputs["BARO"] = SensorPacket("BARO_B", 0.0, np.array([101.325]), healthy=True)
    
    tmr.nodes[2].sensor_inputs["IMU"] = SensorPacket("IMU_C", 0.0, np.array([0., 0., 9.8, 0., 0., 0.]), healthy=True)
    tmr.nodes[2].sensor_inputs["BARO"] = SensorPacket("BARO_C", 0.0, np.array([101.325]), healthy=True)
    
    voted_cmd, mode = tmr.step()
    assert tmr.nodes[0].sensor_inputs["BARO"].healthy is True
    assert tmr.nodes[1].sensor_inputs["IMU"].healthy is True
    assert mode == SystemMode.TRIPLEX


def test_apollo_tmr_all_voting_branches_and_degradation():
    tmr = ApolloTMRFlightExecutive(tolerance=0.05, max_transient_mismatches=2)
    
    # Case: Node 0 outlier (Nodes 1 & 2 agree)
    cmds = [np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.5])]
    voted, agreeing, diverging = tmr._vote_majority(cmds)
    assert agreeing == [1, 2]
    assert diverging == [0]
    
    # Case: Node 1 outlier (Nodes 0 & 2 agree)
    cmds = [np.array([0.5, 0.5, 0.5, 0.5]), np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5, 0.5])]
    voted, agreeing, diverging = tmr._vote_majority(cmds)
    assert agreeing == [0, 2]
    assert diverging == [1]
    
    # Case: All 3 mutually disagree -> SAFE_HOLD
    cmds = [np.array([0.1, 0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.9, 0.9, 0.9, 0.9])]
    voted, agreeing, diverging = tmr._vote_majority(cmds)
    assert tmr.current_mode == SystemMode.SAFE_HOLD
    assert len(agreeing) == 0
    
    # Case: 2 active nodes agree -> DUPLEX mode
    tmr.nodes[2].health = NodeHealth.HARD_FAULT
    cmds = [np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.51, 0.51, 0.51, 0.51]), np.zeros(4)]
    voted, agreeing, diverging = tmr._vote_majority(cmds)
    assert tmr.current_mode == SystemMode.DUPLEX
    assert agreeing == [0, 1]
    
    # Case: 2 active nodes disagree -> SAFE_HOLD
    cmds = [np.array([0.1, 0.1, 0.1, 0.1]), np.array([0.9, 0.9, 0.9, 0.9]), np.zeros(4)]
    voted, agreeing, diverging = tmr._vote_majority(cmds)
    assert tmr.current_mode == SystemMode.SAFE_HOLD
    
    # Case: 1 active node -> SIMPLEX mode
    tmr.nodes[1].health = NodeHealth.HARD_FAULT
    cmds = [np.array([0.6, 0.6, 0.6, 0.6]), np.zeros(4), np.zeros(4)]
    voted, agreeing, diverging = tmr._vote_majority(cmds)
    assert tmr.current_mode == SystemMode.SIMPLEX
    assert agreeing == [0]
    
    # Case: 0 active nodes -> SAFE_HOLD
    tmr.nodes[0].health = NodeHealth.HARD_FAULT
    voted, agreeing, diverging = tmr._vote_majority(cmds)
    assert tmr.current_mode == SystemMode.SAFE_HOLD


def test_apollo_tmr_hard_fault_isolation_state_machine():
    tmr = ApolloTMRFlightExecutive(tolerance=0.05, max_transient_mismatches=2)
    
    # Node 1 starts HEALTHY and experiences 1 divergence -> becomes TRANSIENT_ANOMALY (line 312)
    tmr.nodes[1].health = NodeHealth.HEALTHY
    tmr.nodes[1].consecutive_mismatches = 0
    tmr._fdir_reconcile(np.zeros(4), agreeing_nodes=[0, 2], diverging_nodes=[1])
    assert tmr.nodes[1].health == NodeHealth.TRANSIENT_ANOMALY
    assert tmr.nodes[1].consecutive_mismatches == 1
    
    # Consecutive second mismatch -> becomes HARD_FAULT
    tmr._fdir_reconcile(np.zeros(4), agreeing_nodes=[0, 2], diverging_nodes=[1])
    assert tmr.nodes[1].health == NodeHealth.HARD_FAULT
    assert any(ev["event"] == "HARD_FAULT_ISOLATION" for ev in tmr.fdir_events_log)
    
    # Agreeing node error decrement
    tmr.nodes[0].consecutive_mismatches = 2
    tmr._fdir_reconcile(np.zeros(4), agreeing_nodes=[0, 2], diverging_nodes=[])
    assert tmr.nodes[0].consecutive_mismatches == 1


def test_flight_node_hard_fault_execution():
    node = FlightComputerNode("TestNode")
    node.health = NodeHealth.HARD_FAULT
    cmd = node.execute_control_cycle(step=1, dt=0.01)
    assert len(cmd) == 4
    assert np.any(cmd > 100.0)
