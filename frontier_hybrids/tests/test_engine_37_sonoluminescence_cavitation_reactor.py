from engine_37_sonoluminescence_cavitation_reactor import SonoluminescenceReactorEngine

def test_sonoluminescence():
    engine = SonoluminescenceReactorEngine(seed=42)
    res = engine.step_cavitation(acoustic_pressure_kpa=140.0, dt_ns=0.1)
    assert res["bubble_radius_um"] > 0.0
    assert res["core_temperature_k"] >= 300.0
