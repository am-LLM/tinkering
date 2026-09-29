from engine_21_metamaterial_cloaking_rf_harvester import MetamaterialCloakRFHarvesterEngine

def test_metamaterial_harvester():
    engine = MetamaterialCloakRFHarvesterEngine(seed=42)
    res = engine.step_harvest(rf_frequency_ghz=5.8)
    assert res["harvested_power_mw"] > 0.0
    assert res["scattering_cross_section_db"] < -20.0
