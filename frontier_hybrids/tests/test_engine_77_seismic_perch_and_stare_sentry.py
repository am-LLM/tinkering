import pytest
import numpy as np
from frontier_hybrids.engine_77_seismic_perch_and_stare_sentry import DormantSeismicSentryEngine

def test_engine_77_initialization():
    engine = DormantSeismicSentryEngine(battery_capacity_mAh=500.0)
    assert engine.current_state == "DORMANT_DEEP_SLEEP"
    # Quiescent current of 12uA should yield over 1000 days of sleep life
    assert engine.compute_projected_battery_life_days(duty_cycle_pct=0.0) > 1000.0

def test_engine_77_dormant_sleep_maintenance():
    engine = DormantSeismicSentryEngine()
    
    # Ambient baseline quiet
    res = engine.process_sentry_tick(
        time_elapsed_hours=4.0,
        piezo_accel_g=0.01,
        mic_energy_db=32.0,
        ambient_db=30.0
    )
    
    assert res["sentry_state"] == "DORMANT_DEEP_SLEEP"
    assert res["wake_trigger_fired"] is False
    assert res["propulsion_armed"] is False
    assert res["hours_in_hibernation"] == 4.0

def test_engine_77_seismic_vehicle_awakening():
    engine = DormantSeismicSentryEngine()
    
    # Passing heavy vehicle produces seismic shock > 80mg and low-frequency rumble
    res = engine.process_sentry_tick(
        time_elapsed_hours=0.01,
        piezo_accel_g=0.15,
        mic_energy_db=48.0,
        ambient_db=32.0
    )
    
    assert res["sentry_state"] == "ACTIVE_AWAKE"
    assert res["wake_trigger_fired"] is True
    assert res["trigger_reason"] == "DUAL_SEISMIC_ACOUSTIC_CONFIRMATION"
    assert res["propulsion_armed"] is True
    assert res["optical_camera_powered"] is True
