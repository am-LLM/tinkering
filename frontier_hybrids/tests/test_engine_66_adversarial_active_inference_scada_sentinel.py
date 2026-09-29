"""
Unit Tests for Engine 66: Adversarial Active Inference SCADA Sentinel
"""

import numpy as np
import pytest
from engine_66_adversarial_active_inference_scada_sentinel import ActiveInferenceScadaSentinel

def test_nominal_tracking():
    sentinel = ActiveInferenceScadaSentinel()
    
    # Feed nominal physical trajectory
    for step in range(50):
        t = step * 0.01
        true_state = np.array([np.sin(t), np.cos(t), 0.1 * np.sin(2*t), 0.05 * np.cos(t)])
        noise = np.random.normal(0, 0.01, size=4)
        obs = true_state + noise
        
        res = sentinel.step(obs)
        assert res["alert_level"] in ["NOMINAL", "ATTACK_DETECTED"]
        assert not res["rollback_triggered"]

def test_fdia_stealth_sensor_spoofing_quarantine():
    sentinel = ActiveInferenceScadaSentinel()
    
    # Initialize with steady state
    for _ in range(20):
        sentinel.step(np.zeros(4))
        
    # Inject stealth False Data Injection Attack on channel 1 (within nominal min/max static range, but violating dynamics)
    spoofed_obs = np.array([0.0, 1.2, 0.0, 0.0])  # Discontinuous velocity jump
    res = sentinel.step(spoofed_obs)
    
    assert res["alert_level"] == "ATTACK_DETECTED"
    assert 1 in res["quarantined_channels"]
    assert res["free_energy"] > 5.0

def test_emergency_state_rollback():
    sentinel = ActiveInferenceScadaSentinel()
    
    # Establish valid baseline
    sentinel.step(np.array([0.1, 0.1, 0.0, 0.0]))
    
    # Inject massive catastrophic actuator manipulation
    extreme_obs = np.array([15.0, 25.0, 20.0, 20.0])
    res = sentinel.step(extreme_obs)
    
    # Verify rollback triggers to protect physical plant
    assert res["rollback_triggered"]
    assert res["alert_level"] == "EMERGENCY_ROLLBACK_EXECUTED"
    assert np.all(np.abs(res["filtered_state_belief"]) <= 2.0)
