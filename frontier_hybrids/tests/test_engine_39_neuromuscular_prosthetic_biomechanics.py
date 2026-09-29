from engine_39_neuromuscular_prosthetic_biomechanics import HillMuscleProstheticEngine

def test_hill_prosthetic():
    engine = HillMuscleProstheticEngine(seed=42)
    res = engine.step_prosthetic_actuation(emg_signal_mv=1.5, dt=0.02)
    assert res["muscle_force_n"] >= 0.0
    assert 0.0 <= res["joint_angle_deg"] <= 120.0
