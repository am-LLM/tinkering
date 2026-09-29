from engine_02_neuromorphic_jump_diffusion import NeuromorphicJumpDiffusionEngine

def test_neuromorphic_jump_diffusion():
    engine = NeuromorphicJumpDiffusionEngine(num_neurons=16, initial_capital=100.0, seed=42)
    assert engine.capital == 100.0
    for _ in range(30):
        out = engine.step(dt=0.01)
        assert out["capital"] >= 0.0
        assert 0.0 <= out["firing_rate"] <= 1.0
