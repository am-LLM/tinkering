import numpy as np
from engine_43_spintronic_mram_neuromorphic_crossbar import SpintronicMRAMNeuromorphicEngine

def test_spintronic_mram():
    engine = SpintronicMRAMNeuromorphicEngine(size=4, seed=42)
    pre = np.array([10.0, 15.0, 20.0, 25.0])
    post = np.array([12.0, 18.0, 19.0, 30.0])
    res = engine.step_stdp_update(pre, post)
    assert 0.0 <= res["mean_weight"] <= 1.0
    assert res["total_conductance_us"] > 0.0
