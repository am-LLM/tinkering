from bulletproofs_inner_product_range import BulletproofsInnerProductRange
import numpy as np

def test_bulletproofsinnerproductrange_response():
    engine = BulletproofsInnerProductRange(nominal_scale=2.0, channels=4)
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    resp = engine.compute_response(sig)
    assert len(resp) == 4
    assert np.all(np.isfinite(resp))
    assert engine.energy_metric() > 0.0

def test_bulletproofsinnerproductrange_step():
    engine = BulletproofsInnerProductRange(nominal_scale=1.5, channels=3)
    val = engine.step_simulation(dt=0.05)
    assert np.isfinite(val)
