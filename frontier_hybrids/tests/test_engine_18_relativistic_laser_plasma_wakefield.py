from engine_18_relativistic_laser_plasma_wakefield import LaserPlasmaWakefieldEngine

def test_laser_plasma_wakefield():
    engine = LaserPlasmaWakefieldEngine(seed=42)
    res = engine.step_acceleration(propagation_distance_cm=5.0)
    assert res["beam_energy_gev"] > 0.0
    assert res["accelerating_gradient_gv_m"] > 0.0
