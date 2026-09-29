from engine_49_neuromorphic_auditory_sound_localizer import JeffressSoundLocalizerEngine

def test_sound_localizer():
    engine = JeffressSoundLocalizerEngine(seed=42)
    left, right = engine.compute_binaural_signals(true_azimuth_deg=30.0, freq_hz=500.0)
    res = engine.localize_sound_azimuth(left, right)
    assert abs(res["estimated_azimuth_deg"] - 30.0) < 15.0
