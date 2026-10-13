import pytest
import numpy as np
from frontier_hybrids.engine_75_visual_servoing_terminal_pro_nav import VisualServoingTerminalNavEngine

def test_engine_75_initialization():
    engine = VisualServoingTerminalNavEngine(nav_constant_n=3.5, max_g_load=4.0)
    assert engine.nav_n == 3.5
    assert engine.max_g == 4.0
    assert engine.is_target_locked is False

def test_engine_75_pro_nav_guidance():
    engine = VisualServoingTerminalNavEngine()
    engine.lock_optical_target((480, 640), np.array([320.0, 240.0, 40.0, 40.0]))
    
    drone_pos = np.array([0.0, 0.0, 50.0])
    drone_vel = np.array([25.0, 0.0, 0.0]) # Forward flight at 25 m/s
    
    # Target 300m ahead moving perpendicularly to the right
    target_pos = np.array([300.0, 20.0, 0.0])
    target_vel = np.array([0.0, 10.0, 0.0])
    
    res = engine.execute_terminal_guidance_step(drone_pos, drone_vel, target_pos, target_vel)
    assert res["terminal_lock_engaged"] is True
    assert res["applied_g_load"] <= 4.0 # Enforces max G limit
    assert res["guidance_state"] == "TERMINAL_PRO_NAV_ACTIVE"

def test_engine_75_cbf_high_g_enforcement():
    engine = VisualServoingTerminalNavEngine(max_g_load=2.5) # Strict 2.5G limit
    
    # Target maneuvering aggressively causing high LOS turn rate
    drone_pos = np.array([0.0, 0.0, 20.0])
    drone_vel = np.array([40.0, 0.0, 0.0])
    
    target_pos = np.array([40.0, 30.0, 20.0])
    target_vel = np.array([0.0, 35.0, 0.0])
    
    res = engine.execute_terminal_guidance_step(drone_pos, drone_vel, target_pos, target_vel)
    # CBF must clamp commanded acceleration to 2.5G
    assert res["cbf_g_limit_enforced"] is True
    assert np.isclose(res["applied_g_load"], 2.5, atol=1e-2)
