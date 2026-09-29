from engine_26_magnetohydrodynamic_molten_salt_pump import MHDMoltenSaltPumpEngine

def test_mhd_pump():
    engine = MHDMoltenSaltPumpEngine(seed=42)
    res = engine.step_pumping(loop_pressure_drop_pa=1.0e4, dt=0.01)
    assert res["flow_velocity_m_s"] >= 0.0
    assert res["lorentz_pressure_pa"] > 0.0
