import numpy as np
from aeroelastic_galloping_oscillator import AeroelasticFlutterModel

def test_aeroelastic_flutter_sim():
    model = AeroelasticFlutterModel()
    alphas = [model.step(airspeed_u=15.0) for _ in range(200)]
    assert len(alphas) == 200
    assert not any(np.isnan(alphas))
