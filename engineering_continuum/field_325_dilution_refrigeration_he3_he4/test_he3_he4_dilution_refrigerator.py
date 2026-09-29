from he3_he4_dilution_refrigerator import He3He4DilutionRefrigerator
import numpy as np

def test_he3he4dilutionrefrigerator_response():
    engine = He3He4DilutionRefrigerator(nominal_scale=2.0, channels=4)
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    resp = engine.compute_response(sig)
    assert len(resp) == 4
    assert np.all(np.isfinite(resp))
    assert engine.energy_metric() > 0.0

def test_he3he4dilutionrefrigerator_step():
    engine = He3He4DilutionRefrigerator(nominal_scale=1.5, channels=3)
    val = engine.step_simulation(dt=0.05)
    assert np.isfinite(val)
