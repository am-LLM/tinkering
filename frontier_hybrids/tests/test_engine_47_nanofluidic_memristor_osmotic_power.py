from engine_47_nanofluidic_memristor_osmotic_power import NanofluidicOsmoticPowerEngine

def test_osmotic_power():
    engine = NanofluidicOsmoticPowerEngine(seed=42)
    res = engine.step_blue_energy_harvest(c_sea_molar=0.5, c_river_molar=0.01)
    assert res["osmotic_voltage_mv"] > 0.0
    assert res["power_density_w_m2"] > 0.0
