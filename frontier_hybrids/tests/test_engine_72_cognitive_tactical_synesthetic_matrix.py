import pytest
import numpy as np
from frontier_hybrids.engine_72_cognitive_tactical_synesthetic_matrix import CognitiveTacticalSynestheticEngine

def test_engine_72_initialization():
    engine = CognitiveTacticalSynestheticEngine(tongue_grid_dim=8)
    assert engine.tongue_matrix.shape == (8, 8)
    assert engine.hrv_baseline_ms == 55.0

def test_engine_72_cognitive_tunneling_diversion():
    engine = CognitiveTacticalSynestheticEngine()
    
    # Low HRV (stressed), High ACC conflict, high beta -> severe tunneling
    hrv = 18.0
    acc_conflict = 0.85
    beta = 0.70
    
    eeg = np.random.normal(0, 0.1, (2, 250)) # No P300
    threats = [
        {"id": "SAM_RADAR_01", "azimuth_deg": 90.0, "distance_m": 800.0, "severity": 0.9},
        {"id": "AIR_INTERCEPT_02", "azimuth_deg": 180.0, "distance_m": 500.0, "severity": 1.0}
    ]
    
    res = engine.process_synesthetic_tactical_step(hrv, acc_conflict, beta, eeg, threats)
    assert res["cognitive_tunneling_detected"] is True
    assert res["sensory_substitution_active"] is True
    assert res["lingual_matrix_density"] > 0.0
    assert "COGNITIVE TUNNELING ACTIVE" in res["doctrine_directive"]

def test_engine_72_p300_subconscious_locking():
    engine = CognitiveTacticalSynestheticEngine()
    
    # Calm state
    hrv = 60.0
    acc_conflict = 0.1
    beta = 0.2
    
    # Inject P300 spike in samples 115-140
    eeg = np.random.normal(0, 0.05, (2, 250))
    eeg[0, 115:140] += 1.8 # Strong Pz peak
    
    threats = [{"id": "STEALTH_CFAR_BOGEY", "azimuth_deg": 15.0, "distance_m": 1200.0, "severity": 0.8}]
    
    res = engine.process_synesthetic_tactical_step(hrv, acc_conflict, beta, eeg, threats)
    assert res["p300_subconscious_target_lock"] is True
    assert res["active_locked_target_id"] == "STEALTH_CFAR_BOGEY"
    assert res["sensory_substitution_active"] is False
