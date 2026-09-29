from engine_03_quantum_annealing_haptics import QuantumAnnealingHapticsEngine

def test_quantum_haptics():
    engine = QuantumAnnealingHapticsEngine(num_actuators=8, seed=42)
    res = engine.compute_haptic_feedback()
    assert res["mean_force_feedback"] >= 0.0
    assert "energy" in res
