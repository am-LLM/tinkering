"""
Unit Tests for Z3 Formal Swarm Deconfliction: SMT Mathematical Proofs for 100+ Agents
"""

import pytest
import z3
import numpy as np
from unittest.mock import patch, MagicMock
from z3_formal_swarm_deconflict import Z3SwarmFormalVerifier, SwarmVerificationConfig


def test_theorem1_pairwise_collision_impossibility():
    verifier = Z3SwarmFormalVerifier(SwarmVerificationConfig(r_safe=2.0, dt=0.5))
    
    p1 = (0.0, 0.0)
    v1 = (1.0, 0.0)
    p2 = (10.0, 0.0)
    v2 = (-1.0, 0.0)
    
    is_safe, status = verifier.verify_pairwise_collision_impossibility(p1, v1, p2, v2)
    assert is_safe is True
    assert status == "FORMALLY_PROVEN_UNSAT_NO_COLLISION"


def test_theorem1_collision_counterexample_detection():
    verifier = Z3SwarmFormalVerifier(SwarmVerificationConfig(r_safe=2.0, dt=1.0))
    
    p1 = (0.0, 0.0)
    v1 = (5.0, 0.0)
    p2 = (4.0, 0.0)
    v2 = (-5.0, 0.0)
    
    is_safe, status = verifier.verify_pairwise_collision_impossibility(p1, v1, p2, v2)
    assert is_safe is False
    assert "COUNTEREXAMPLE" in status


def test_theorem1_unknown_solver_state():
    verifier = Z3SwarmFormalVerifier(SwarmVerificationConfig(r_safe=2.0, dt=1.0))
    with patch.object(z3.Solver, "check", return_value=z3.unknown):
        is_safe, status = verifier.verify_pairwise_collision_impossibility((0, 0), (0, 0), (1, 1), (0, 0))
        assert is_safe is False
        assert status == "UNKNOWN_SOLVER_STATE"


def test_theorem2_deadlock_freedom():
    verifier = Z3SwarmFormalVerifier(SwarmVerificationConfig(r_safe=2.0, v_max=5.0, dt=0.5))
    
    p_curr = (0.0, 0.0)
    p_goal = (10.0, 0.0)
    obstacles = [(5.0, 0.0)]
    
    has_liveness, v_sol = verifier.verify_deadlock_freedom(p_curr, p_goal, obstacles)
    assert has_liveness is True
    assert v_sol is not None
    vx, vy = v_sol
    assert vx > 0.0 or vy != 0.0


def test_theorem2_deadlock_impossibility_case():
    verifier = Z3SwarmFormalVerifier(SwarmVerificationConfig(r_safe=5.0, v_max=2.0, dt=0.5))
    p_curr = (0.0, 0.0)
    p_goal = (10.0, 0.0)
    dense_obstacles = [
        (0.5, 0.0), (0.5, 0.5), (0.5, -0.5),
        (0.0, 0.5), (0.0, -0.5), (0.3, 0.3), (0.3, -0.3),
        (1.0, 0.0), (1.0, 0.5), (1.0, -0.5)
    ]
    has_liveness, v_sol = verifier.verify_deadlock_freedom(p_curr, p_goal, dense_obstacles)
    assert has_liveness is False
    assert v_sol is None


def test_theorem3_scalable_100_agent_swarm_verification():
    verifier = Z3SwarmFormalVerifier(SwarmVerificationConfig(r_safe=2.0, dt=0.5))
    
    num_agents = 120
    agent_states = []
    
    grid_side = 12
    spacing = 5.0
    
    for i in range(num_agents):
        row = i // grid_side
        col = i % grid_side
        px = col * spacing
        py = row * spacing
        
        vx = 1.0 if row % 2 == 0 else -1.0
        vy = 0.0
        
        agent_states.append({
            "id": i,
            "p": (px, py),
            "v": (vx, vy)
        })
        
    result = verifier.verify_large_scale_swarm_safety(agent_states)
    assert result["total_agents"] == 120
    assert result["proven_safe_pairs"] > 0
    assert result["counterexamples_count"] == 0
    assert result["global_safety_formally_proven"] is True


def test_theorem3_scalable_swarm_with_counterexample_and_shared_cells():
    verifier = Z3SwarmFormalVerifier(SwarmVerificationConfig(r_safe=2.0, dt=1.0))
    agent_states = [
        {"id": 1, "p": (0.0, 0.0), "v": (5.0, 0.0)},
        {"id": 2, "p": (4.0, 0.0), "v": (-5.0, 0.0)},
        {"id": 3, "p": (0.5, 0.5), "v": (0.0, 0.0)},
        {"id": 4, "p": (50.0, 50.0), "v": (0.0, 0.0)}
    ]
    result = verifier.verify_large_scale_swarm_safety(agent_states)
    assert result["global_safety_formally_proven"] is False
    assert result["counterexamples_count"] > 0
