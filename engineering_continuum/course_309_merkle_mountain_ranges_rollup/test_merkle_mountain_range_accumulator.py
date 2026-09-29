from merkle_mountain_range_accumulator import MerkleMountainRangeAccumulator
import numpy as np

def test_merklemountainrangeaccumulator_response():
    engine = MerkleMountainRangeAccumulator(nominal_scale=2.0, channels=4)
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    resp = engine.compute_response(sig)
    assert len(resp) == 4
    assert np.all(np.isfinite(resp))
    assert engine.energy_metric() > 0.0

def test_merklemountainrangeaccumulator_step():
    engine = MerkleMountainRangeAccumulator(nominal_scale=1.5, channels=3)
    val = engine.step_simulation(dt=0.05)
    assert np.isfinite(val)
