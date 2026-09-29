from engine_14_geodesic_solar_sail_nav import GeodesicSolarSailNavEngine

def test_solar_sail_nav():
    engine = GeodesicSolarSailNavEngine(seed=42)
    res = engine.step_orbital_nav(sail_pitch_deg=35.0, dt_days=0.5)
    assert res["heliocentric_distance_au"] > 0.0
    assert res["orbital_speed_kms"] > 0.0
