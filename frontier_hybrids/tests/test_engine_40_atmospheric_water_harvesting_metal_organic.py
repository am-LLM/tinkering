from engine_40_atmospheric_water_harvesting_metal_organic import AtmosphericWaterMOFEngine

def test_mof_water_harvester():
    engine = AtmosphericWaterMOFEngine(mof_mass_kg=2.0, seed=42)
    # Night adsorption
    engine.step_harvest_cycle(ambient_rh_pct=60.0, solar_irradiance_w_m2=0.0, dt_hours=4.0)
    assert engine.water_uptake_g_g > 0.1
    # Day solar desorption
    res = engine.step_harvest_cycle(ambient_rh_pct=30.0, solar_irradiance_w_m2=850.0, dt_hours=2.0)
    assert res["desorbed_liters_step"] > 0.0
