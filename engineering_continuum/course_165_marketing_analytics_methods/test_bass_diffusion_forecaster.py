from bass_diffusion_forecaster import BassDiffusionForecaster

def test_bass():
    model = BassDiffusionForecaster()
    a1 = model.forecast_period(0.0)
    assert a1 == 3000.0
