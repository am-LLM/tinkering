from engine_38_magnetostrictive_sonar_transducer import MagnetostrictiveSonarEngine

def test_magnetostrictive_sonar():
    engine = MagnetostrictiveSonarEngine(seed=42)
    res = engine.step_transmit_pulse(coil_current_a=10.0, water_depth_m=1000.0)
    assert res["rod_displacement_um"] > 0.0
    assert res["source_level_db"] > 150.0
