from engine_30_stochastic_resonance_sensor_amplifier import StochasticResonanceMEMSEngine

def test_stochastic_resonance():
    engine = StochasticResonanceMEMSEngine(seed=42)
    res = engine.step_resonance(sub_threshold_signal=0.05, noise_intensity_d=0.3, dt=0.01)
    assert "output_state_x" in res
    assert res["kramers_escape_rate"] > 0.0
