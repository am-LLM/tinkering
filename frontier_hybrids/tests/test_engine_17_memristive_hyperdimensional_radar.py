from engine_17_memristive_hyperdimensional_radar import MemristiveHyperdimensionalRadarEngine

def test_memristive_radar():
    engine = MemristiveHyperdimensionalRadarEngine(dim=256, seed=42)
    vec = engine.encode_sar_feature("TANK", snr_noise=0.1)
    res = engine.classify_target(vec)
    assert res["target_detected"] == 1.0
