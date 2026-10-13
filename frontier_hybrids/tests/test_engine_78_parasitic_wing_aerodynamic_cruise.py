import pytest
import numpy as np
from frontier_hybrids.engine_78_parasitic_wing_aerodynamic_cruise import ParasiticWingTransitionEngine

def test_engine_78_initialization():
    engine = ParasiticWingTransitionEngine()
    assert engine.current_flight_mode == "HOVER_VERTICAL"
    assert engine.min_cruise_speed_m_s == 8.5

def test_engine_78_hover_mode_full_power():
    engine = ParasiticWingTransitionEngine()
    
    # In pure zero-speed hover, power is ~75W and savings are ~0%
    res = engine.process_transition_step(commanded_speed_m_s=0.0, pitch_angle_deg=0.0)
    assert res["flight_mode"] == "HOVER_VERTICAL"
    assert np.isclose(res["electrical_power_watts"], 75.0, atol=1.0)
    assert res["cruising_range_multiplier"] <= 1.1

def test_engine_78_aerodynamic_cruise_energy_savings():
    engine = ParasiticWingTransitionEngine()
    
    # In forward aerodynamic cruise at 12 m/s, wing produces lift
    res = engine.process_transition_step(commanded_speed_m_s=12.0, pitch_angle_deg=5.0)
    assert res["flight_mode"] == "FORWARD_AERODYNAMIC_CRUISE"
    assert res["energy_savings_pct"] > 40.0 # Significant reduction in power
    assert res["cruising_range_multiplier"] > 1.8 # Extended range
    assert res["cbf_stall_envelope_safe"] is True
