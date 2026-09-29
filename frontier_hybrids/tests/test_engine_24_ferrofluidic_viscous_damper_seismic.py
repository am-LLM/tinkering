from engine_24_ferrofluidic_viscous_damper_seismic import FerrofluidSeismicEngine

def test_ferrofluid_seismic():
    engine = FerrofluidSeismicEngine(seed=42)
    res = engine.step_seismic(ground_accel_ms2=2.5, dt=0.005)
    assert "building_displacement_mm" in res
    assert res["effective_viscosity"] > 0.0
