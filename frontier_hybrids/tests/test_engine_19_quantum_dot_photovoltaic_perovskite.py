from engine_19_quantum_dot_photovoltaic_perovskite import QuantumDotPerovskiteEngine

def test_qd_perovskite():
    engine = QuantumDotPerovskiteEngine(seed=42)
    res = engine.step_photovoltaic_conversion(solar_irradiance_w_m2=1000.0)
    assert 10.0 <= res["power_conversion_efficiency_percent"] <= 40.0
    assert res["power_output_w_m2"] > 0.0
