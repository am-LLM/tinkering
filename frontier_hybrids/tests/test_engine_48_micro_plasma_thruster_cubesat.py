from engine_48_micro_plasma_thruster_cubesat import HeliconPlasmaThrusterEngine

def test_helicon_thruster():
    engine = HeliconPlasmaThrusterEngine(seed=42)
    res = engine.step_thrust_vector(magnetic_nozzle_angle_deg=15.0)
    assert res["total_thrust_mn"] > 0.0
    assert res["specific_impulse_sec"] > 500.0
