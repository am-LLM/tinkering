from engine_41_quantum_cascade_laser_spectroscopy import QuantumCascadeSpectroscopyEngine

def test_qcl_spectroscopy():
    engine = QuantumCascadeSpectroscopyEngine(seed=42)
    res = engine.detect_trace_gas(trace_gas_ppm=25.0, laser_power_mw=20.0)
    assert res["transmitted_power_mw"] > 0.0
    assert abs(res["estimated_ppm"] - 25.0) < 5.0
