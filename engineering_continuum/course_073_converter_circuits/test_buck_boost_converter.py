from buck_boost_converter import BuckBoostConverter

def test_buck_boost_metrics():
    conv = BuckBoostConverter()
    assert abs(conv.steady_state_gain(0.5) - 1.0) < 1e-4
