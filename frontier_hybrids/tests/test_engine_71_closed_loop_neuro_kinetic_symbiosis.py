import pytest
import numpy as np
from frontier_hybrids.engine_71_closed_loop_neuro_kinetic_symbiosis import NeuroKineticSymbiosisEngine

def test_engine_71_initialization():
    engine = NeuroKineticSymbiosisEngine(sample_rate=250.0, num_eeg_ch=8, num_emg_ch=4)
    assert engine.sample_rate == 250.0
    assert engine.num_eeg_ch == 8
    assert engine.num_emg_ch == 4
    assert len(engine.control_surfaces) == 4

def test_engine_71_pre_reflex_trimming():
    engine = NeuroKineticSymbiosisEngine()
    
    # Synthetic EEG with lateral asymmetric mu desynchronization (C4 lower than C3 -> right roll intent)
    t = np.arange(250) / 250.0
    eeg = np.ones((8, 250)) * 0.1
    eeg[0, :] += np.sin(2 * np.pi * 10.0 * t) * 2.0 # High C3 mu
    eeg[-1, :] += np.sin(2 * np.pi * 10.0 * t) * 0.2 # Suppressed C4 mu
    
    # Low EMG (pilot has not yet moved muscle)
    emg = np.random.normal(0, 0.02, (4, 250))
    
    telemetry = {"pitch_rad": 0.0, "altitude_m": 100.0, "velocity_z": 0.0}
    res = engine.execute_symbiosis_step(eeg, emg, telemetry)
    
    assert res["pre_trim_active"] is True
    assert res["operational_mode"] == "PRE_REFLEX_TRIM"
    assert res["bci_intent"]["confidence"] > 0.6
    assert res["cbf_invariance_enforced"] is False

def test_engine_71_cbf_safety_intervention():
    engine = NeuroKineticSymbiosisEngine()
    
    # Normal EEG/EMG
    eeg = np.random.normal(0, 0.1, (8, 250))
    emg = np.random.normal(0, 0.3, (4, 250)) # Actuation confirmed
    
    # Dangerous telemetry: low altitude diving rapidly
    telemetry = {"pitch_rad": -0.2, "altitude_m": 8.0, "velocity_z": -3.5}
    res = engine.execute_symbiosis_step(eeg, emg, telemetry)
    
    # CBF must force pitch up to protect altitude invariant
    assert res["cbf_invariance_enforced"] is True
    assert res["actuated_control_surfaces"]["elevator"] >= 0.5
