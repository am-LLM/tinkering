from engine_06_rl_hypersonic_plasma import HypersonicRLMHDControlEngine

def test_hypersonic_rl_mhd():
    engine = HypersonicRLMHDControlEngine(seed=42)
    for _ in range(25):
        res = engine.step(dt=0.05)
    assert res["b_field_tesla"] >= 0.0
    assert res["heat_flux_mw_m2"] < 3.0
