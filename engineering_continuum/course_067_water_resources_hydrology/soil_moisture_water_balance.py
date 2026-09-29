"""Course 067: Thornthwaite-Mather Soil Moisture Deficit & Evapotranspiration Balance"""
class SoilWaterBalance:
    def __init__(self, field_capacity_mm=150.0, initial_soil_moisture_mm=100.0):
        self.fc = field_capacity_mm
        self.moisture = initial_soil_moisture_mm

    def step_month(self, precipitation_mm: float, pet_mm: float) -> dict:
        net_input = precipitation_mm - pet_mm
        recharge = 0.0
        runoff = 0.0
        if net_input >= 0:
            self.moisture += net_input
            if self.moisture > self.fc:
                runoff = self.moisture - self.fc
                self.moisture = self.fc
                recharge = runoff * 0.4
        else:
            self.moisture = max(0.0, self.moisture + net_input)
            
        deficit = max(0.0, self.fc - self.moisture)
        return {
            "soil_moisture_mm": round(float(self.moisture), 2),
            "runoff_mm": round(float(runoff), 2),
            "recharge_mm": round(float(recharge), 2),
            "deficit_mm": round(float(deficit), 2)
        }
