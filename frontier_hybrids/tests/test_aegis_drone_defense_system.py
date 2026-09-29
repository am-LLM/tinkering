"""
Unit Tests for AEGIS Multi-Medium Drone Defense System
"""

import numpy as np
import pytest
from aegis_drone_defense_system import (
    AcousticSensorArrayNetwork,
    UnjammableKineticInterceptor,
    AegisMultiMediumDefenseSystem
)

def test_acoustic_tdoa_triangulation():
    stations = np.array([
        [0.0, 0.0, 0.0],
        [1000.0, 0.0, 0.0],
        [0.0, 1000.0, 0.0],
        [1000.0, 1000.0, 100.0]
    ])
    network = AcousticSensorArrayNetwork(stations=stations, speed_of_sound=343.0)
    
    true_target = np.array([450.0, 520.0, 150.0])
    
    # Calculate exact arrival times
    distances = np.linalg.norm(stations - true_target, axis=1)
    t_arrivals = distances / 343.0
    
    estimated_pos = network.triangulate_threat_tdoa(t_arrivals)
    
    # Verify triangulation accuracy within 10 meters
    error = np.linalg.norm(estimated_pos - true_target)
    assert error < 15.0

def test_kinetic_proportional_navigation_interception():
    # Target loitering drone flying at 50 m/s (Shahed/Geran profile)
    target_start = np.array([1200.0, 800.0, 300.0])
    target_vel = np.array([-45.0, -20.0, 0.0])
    
    interceptor = UnjammableKineticInterceptor(
        init_pos=np.array([0.0, 0.0, 0.0]),
        init_vel=np.array([80.0, 60.0, 30.0]),
        dt=0.01
    )
    
    intercepted = False
    cur_t_pos = target_start.copy()
    
    for step in range(1500):  # 15 seconds
        cur_t_pos += target_vel * 0.01
        res = interceptor.step(cur_t_pos, target_vel)
        if res["intercepted"]:
            intercepted = True
            break
            
    assert intercepted
    assert res["miss_distance"] <= 1.5

def test_aegis_multi_medium_system_pipeline():
    aegis = AegisMultiMediumDefenseSystem()
    
    # Deploy interceptor
    idx = aegis.deploy_interceptor(
        launch_pos=np.array([100.0, 100.0, 0.0]),
        initial_heading=np.array([0.6, 0.6, 0.3])
    )
    
    # Generate incoming target trajectory (Shahed drone or Black Sea USV)
    target_traj = []
    t_pos = np.array([800.0, 800.0, 150.0])
    t_vel = np.array([-30.0, -30.0, -2.0])
    
    for _ in range(800):
        t_pos = t_pos + t_vel * 0.01
        target_traj.append((t_pos.copy(), t_vel.copy()))
        
    outcome = aegis.evaluate_engagement(target_traj, interceptor_idx=idx)
    assert outcome["kinetic_kill"]
    assert outcome["engagement_duration_sec"] < 10.0
