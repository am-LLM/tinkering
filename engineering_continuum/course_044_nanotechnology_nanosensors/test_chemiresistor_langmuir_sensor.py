from chemiresistor_langmuir_sensor import NanomaterialChemiresistor

def test_nanosensor_response():
    sensor = NanomaterialChemiresistor(r_baseline=10000.0)
    for _ in range(50):
        r = sensor.step_exposure(gas_concentration_ppm=50.0)
    assert r > 10000.0
    assert sensor.theta > 0.5
