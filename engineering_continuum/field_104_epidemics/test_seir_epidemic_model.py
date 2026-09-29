import numpy as np
from seir_epidemic_model import SEIREpidemicModel

def test_seir():
    m = SEIREpidemicModel()
    assert m.r0() == 5.0
    s = np.array([9990.0, 10.0, 0.0, 0.0])
    s_next = m.step(s, 0.1)
    assert abs(np.sum(s_next) - 10000.0) < 1e-4
