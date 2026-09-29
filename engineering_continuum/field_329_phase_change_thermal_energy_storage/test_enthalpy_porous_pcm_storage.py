from enthalpy_porous_pcm_storage import EnthalpyPorousPCMStorage
import numpy as np

def test_enthalpyporouspcmstorage_response():
    engine = EnthalpyPorousPCMStorage(nominal_scale=2.0, channels=4)
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    resp = engine.compute_response(sig)
    assert len(resp) == 4
    assert np.all(np.isfinite(resp))
    assert engine.energy_metric() > 0.0

def test_enthalpyporouspcmstorage_step():
    engine = EnthalpyPorousPCMStorage(nominal_scale=1.5, channels=3)
    val = engine.step_simulation(dt=0.05)
    assert np.isfinite(val)
