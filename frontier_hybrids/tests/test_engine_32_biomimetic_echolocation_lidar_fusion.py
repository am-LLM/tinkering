from engine_32_biomimetic_echolocation_lidar_fusion import BiomimeticBatLiDARFusionEngine

def test_bat_lidar_fusion():
    engine = BiomimeticBatLiDARFusionEngine(seed=42)
    res = engine.step_fusion()
    assert res["mean_ranging_error_m"] < 0.5
    assert res["num_targets_tracked"] == 5.0
