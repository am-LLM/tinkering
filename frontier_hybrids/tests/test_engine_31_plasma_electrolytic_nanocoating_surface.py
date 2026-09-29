from engine_31_plasma_electrolytic_nanocoating_surface import PlasmaElectrolyticCoatingEngine

def test_plasma_coating():
    engine = PlasmaElectrolyticCoatingEngine(seed=42)
    for _ in range(15):
        res = engine.step_deposition(current_density_a_dm2=20.0, dt_sec=1.0)
    assert res["thickness_um"] > 0.0
    assert res["alpha_phase_fraction"] >= 0.1
