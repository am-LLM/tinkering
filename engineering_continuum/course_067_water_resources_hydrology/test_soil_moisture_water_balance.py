from soil_moisture_water_balance import SoilWaterBalance

def test_water_balance():
    model = SoilWaterBalance(field_capacity_mm=150.0, initial_soil_moisture_mm=100.0)
    res_dry = model.step_month(precipitation_mm=20.0, pet_mm=80.0)
    assert res_dry["deficit_mm"] > 50.0
    res_wet = model.step_month(precipitation_mm=200.0, pet_mm=50.0)
    assert res_wet["soil_moisture_mm"] == 150.0
    assert res_wet["runoff_mm"] > 0.0
