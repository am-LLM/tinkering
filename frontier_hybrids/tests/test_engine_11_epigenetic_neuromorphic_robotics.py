from engine_11_epigenetic_neuromorphic_robotics import EpigeneticSoftRoboticsEngine

def test_epigenetic_soft_robotics():
    engine = EpigeneticSoftRoboticsEngine(num_segments=4, seed=42)
    res = engine.step_epigenetic_adaptation(payload_stress=8.0, dt=0.1)
    assert 0.0 <= res["mean_methylation"] <= 1.0
    assert "tip_deflection_m" in res
