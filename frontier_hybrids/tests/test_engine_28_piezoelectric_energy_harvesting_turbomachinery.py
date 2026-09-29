from engine_28_piezoelectric_energy_harvesting_turbomachinery import PiezoTurbomachineryEngine

def test_piezo_turbomachinery():
    engine = PiezoTurbomachineryEngine(blade_pass_freq_hz=1000.0, seed=42)
    res = engine.step_vibration_harvest(vibration_amplitude_g=10.0, duration_ms=20.0)
    assert res["harvested_power_mw"] > 0.0
    assert "stored_energy_uj" in res
