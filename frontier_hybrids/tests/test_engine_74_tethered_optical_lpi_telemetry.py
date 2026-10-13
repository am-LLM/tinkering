import pytest
import numpy as np
from frontier_hybrids.engine_74_tethered_optical_lpi_telemetry import TetheredOpticalLPIEngine

def test_engine_74_initialization():
    engine = TetheredOpticalLPIEngine(max_spool_length_m=5000.0, tensile_strength_newtons=12.0)
    assert engine.max_spool_length_m == 5000.0
    assert engine.optical_link_up is True
    assert engine.deployed_length_m == 0.0

def test_engine_74_normal_payout():
    engine = TetheredOpticalLPIEngine(max_spool_length_m=5000.0, tensile_strength_newtons=15.0)
    
    pos = np.array([500.0, 500.0, 50.0]) # ~700m away
    vel = np.array([12.0, 10.0, 0.0])
    
    res = engine.process_optical_telemetry_frame(tx_packet_bytes=1024, drone_pos=pos, drone_vel=vel)
    assert res["optical_link_active"] is True
    assert res["deployed_tether_m"] > 700.0
    assert res["cbf_strain_intervention"] is False
    assert res["electronic_emission_signature_dbm"] == -999.0

def test_engine_74_cbf_rupture_prevention():
    engine = TetheredOpticalLPIEngine(tensile_strength_newtons=10.0)
    
    # Simulate high aerodynamic drag at long distance and high velocity
    pos = np.array([4000.0, 0.0, 200.0])
    vel = np.array([55.0, 0.0, 15.0]) # Excessive speed causing tension spike
    
    res = engine.process_optical_telemetry_frame(tx_packet_bytes=1024, drone_pos=pos, drone_vel=vel)
    
    # Tension should exceed 80% threshold triggering CBF speed damping
    assert res["cbf_strain_intervention"] is True
    assert res["safe_commanded_velocity"][0] < vel[0] # Forward velocity clamped
    assert res["safe_commanded_velocity"][2] <= 0.0   # Climb arrested
