from engine_35_cellular_automata_turbulent_flame import LatticeBoltzmannFlameEngine

def test_flame_combustion():
    engine = LatticeBoltzmannFlameEngine(grid_size=12, seed=42)
    res = engine.step_combustion(dt=0.1)
    assert 0.0 <= res["burnt_volume_fraction"] <= 1.0
    assert res["turbulent_flame_speed_m_s"] > 0.0
