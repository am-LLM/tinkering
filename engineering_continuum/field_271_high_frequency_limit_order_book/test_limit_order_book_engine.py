from limit_order_book_engine import LimitOrderBookEngine
import numpy as np

def test_limitorderbookengine_metric():
    obj = LimitOrderBookEngine(param=2.5, dimension=3)
    val = obj.compute_metric([1.0, 2.0, 3.0])
    assert np.isclose(val, 2.5 * (1 + 4 + 9))

def test_limitorderbookengine_dynamics():
    obj = LimitOrderBookEngine(param=1.0, dimension=4)
    for _ in range(20):
        st = obj.step_dynamics(dt=0.1)
    assert obj.evaluate_invariant()
    assert np.all(st > 0.0)
