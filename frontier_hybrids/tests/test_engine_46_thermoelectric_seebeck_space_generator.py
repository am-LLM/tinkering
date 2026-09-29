from engine_46_thermoelectric_seebeck_space_generator import ThermoelectricRTGEngine

def test_rtg_generator():
    engine = ThermoelectricRTGEngine(seed=42)
    res = engine.step_mission_years(mission_elapsed_years=10.0, space_sink_temp_k=4.0)
    assert res["electrical_power_w"] > 50.0
    assert res["rtg_efficiency_pct"] > 0.0
