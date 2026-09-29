from engine_42_elastomeric_piezoresistive_electronic_skin import PiezoresistiveElectronicSkinEngine

def test_eskin():
    engine = PiezoresistiveElectronicSkinEngine(num_taxels=8, seed=42)
    res = engine.step_tactile_sense(normal_pressure_kpa=50.0, shear_vibration_hz=250.0, dt=0.01)
    assert res["mean_conductance_ms"] > 0.0
    assert "slip_alarm" in res
