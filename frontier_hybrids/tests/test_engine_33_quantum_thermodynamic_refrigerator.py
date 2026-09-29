from engine_33_quantum_thermodynamic_refrigerator import QuantumRefrigeratorEngine

def test_quantum_refrigerator():
    engine = QuantumRefrigeratorEngine(seed=42)
    res = engine.step_quantum_cooling(drive_power=0.8, dt=0.05)
    assert res["ground_state_population"] > 0.0
    assert res["qubit_coherence_t2_us"] >= 50.0
