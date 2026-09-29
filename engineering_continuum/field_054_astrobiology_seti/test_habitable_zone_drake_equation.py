from habitable_zone_drake_equation import HabitableZoneModel

def test_habitable_zone():
    hz = HabitableZoneModel.calculate_hz_boundaries(stellar_luminosity_solar=1.0)
    assert 0.9 < hz["inner_hz_au"] < 1.0
    assert 1.7 < hz["outer_hz_au"] < 1.8
    n = HabitableZoneModel.drake_equation()
    assert n > 0.0
