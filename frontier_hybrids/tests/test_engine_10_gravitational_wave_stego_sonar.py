from engine_10_gravitational_wave_stego_sonar import GravitationalWaveStegoSonarEngine

def test_gravitational_wave_stego():
    engine = GravitationalWaveStegoSonarEngine(sample_rate_hz=2000, duration_sec=0.2, seed=42)
    template, rx = engine.encode_and_transmit(symbol_bit=1, snr_db=10.0)
    res = engine.matched_filter_decode(template, rx)
    assert res["detected_bit"] == 1.0
    assert res["peak_snr"] > 0.0

    template0, rx0 = engine.encode_and_transmit(symbol_bit=0, snr_db=10.0)
    res0 = engine.matched_filter_decode(template0, rx0)
    assert res0["detected_bit"] == 0.0
