from engine_36_optogenetic_cardiac_defibrillator import OptogeneticCardiacEngine

def test_optogenetic_cardiac():
    engine = OptogeneticCardiacEngine(size=12, seed=42)
    res = engine.step_cardiac_optogenetics(blue_light_flux_mw_mm2=2.5, dt=0.02)
    assert res["mean_action_potential_u"] >= 0.0
    assert "fibrillation_entropy" in res
