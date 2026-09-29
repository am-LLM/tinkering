"""
Unit Tests for Biomimetic Mycelial Swarm Mesh Routing Kernel
"""

import pytest
import numpy as np
from myco_swarm_mesh_routing import (
    MycelialMeshGraph, MycoSwarmRouter, MeshPacket, NodeStatus
)


def test_mycelial_pressure_and_conductance_adaptation():
    graph = MycelialMeshGraph(alpha=1.5, gamma=0.1, mu=1.0)
    
    graph.add_node("Src", pos=np.array([0, 0, 0]))
    graph.add_node("N1", pos=np.array([2, 1, 0]))
    graph.add_node("N2", pos=np.array([2, -5, 0]))
    graph.add_node("Dst", pos=np.array([4, 0, 0]))
    
    graph.add_edge("Src", "N1", length=1.0, initial_conductance=1.0)
    graph.add_edge("N1", "Dst", length=1.0, initial_conductance=1.0)
    
    graph.add_edge("Src", "N2", length=5.0, initial_conductance=1.0)
    graph.add_edge("N2", "Dst", length=5.0, initial_conductance=1.0)
    
    for _ in range(20):
        graph.update_fluxes_and_conductances("Src", "Dst", dt=0.1)
        
    cond_short = graph.edges[("Src", "N1")].conductance
    cond_long = graph.edges[("Src", "N2")].conductance
    assert cond_short > cond_long * 2.0, f"Physarum adaptation failed: short={cond_short}, long={cond_long}"


def test_self_healing_under_50_percent_node_loss():
    graph = MycelialMeshGraph()
    for i in range(6):
        graph.add_node(f"D{i}")
        
    graph.add_edge("D0", "D1", length=1.0, initial_conductance=5.0)
    graph.add_edge("D1", "D3", length=1.0, initial_conductance=5.0)
    graph.add_edge("D3", "D5", length=1.0, initial_conductance=5.0)
    
    graph.add_edge("D0", "D2", length=1.5, initial_conductance=2.0)
    graph.add_edge("D2", "D4", length=1.5, initial_conductance=2.0)
    graph.add_edge("D4", "D5", length=1.5, initial_conductance=2.0)
    
    graph.add_edge("D1", "D4", length=2.0, initial_conductance=1.0)
    
    pkt1 = MeshPacket("P1", "D0", "D5", "telemetry")
    success, path = graph.route_packet(pkt1)
    assert success is True
    assert path == ["D0", "D1", "D3", "D5"]
    
    # INJECT 50% FAILURE: Fail D1 and D3 (primary path destroyed)
    graph.nodes["D1"] = NodeStatus.FAILED
    graph.nodes["D3"] = NodeStatus.FAILED
    
    pkt2 = MeshPacket("P2", "D0", "D5", "emergency_packet")
    success_healed, path_healed = graph.route_packet(pkt2)
    assert success_healed is True
    assert path_healed == ["D0", "D2", "D4", "D5"]


def test_byzantine_blackhole_detection_and_isolation():
    graph = MycelialMeshGraph()
    graph.add_node("A")
    graph.add_node("Malicious_B")
    graph.add_node("Honest_C")
    graph.add_node("D")
    
    # Initial conductances: prefer Malicious_B initially
    graph.add_edge("A", "Malicious_B", length=1.0, initial_conductance=5.0)
    graph.add_edge("Malicious_B", "D", length=1.0, initial_conductance=5.0)
    
    graph.add_edge("A", "Honest_C", length=1.2, initial_conductance=3.0)
    graph.add_edge("Honest_C", "D", length=1.2, initial_conductance=3.0)
    
    # Set blackhole dropping behavior on Malicious_B
    graph.set_blackhole_behavior("Malicious_B", enabled=True)
    
    # Route 5 packets
    for i in range(5):
        pkt = MeshPacket(f"P_{i}", "A", "D", f"data_{i}")
        res, _ = graph.route_packet(pkt)
        assert res is False
        
    # Isolate Byzantine nodes
    isolated = graph.isolate_byzantine_nodes(drop_threshold_ratio=0.3)
    assert "Malicious_B" in isolated
    assert graph.nodes["Malicious_B"] == NodeStatus.BYZANTINE
    
    # Incident edge to Malicious_B should be severed
    assert graph.edges[("A", "Malicious_B")].active is False
    
    # Subsequent packets should now route via Honest_C
    pkt_clean = MeshPacket("P_Clean", "A", "D", "safe_telemetry")
    success, path = graph.route_packet(pkt_clean)
    assert success is True
    assert "Honest_C" in path
    assert "Malicious_B" not in path


def test_myco_swarm_router_end_to_end():
    graph = MycelialMeshGraph()
    for i in range(5):
        graph.add_node(f"Drone_{i}")
    for i in range(4):
        graph.add_edge(f"Drone_{i}", f"Drone_{i+1}", length=1.0)
        
    router = MycoSwarmRouter(graph)
    res = router.send_message("Drone_0", "Drone_4", "MISSION_WAYPOINTS")
    assert res is True
    assert len(router.delivered_packets) == 1
    assert router.delivered_packets[0].hops == ["Drone_0", "Drone_1", "Drone_2", "Drone_3", "Drone_4"]
