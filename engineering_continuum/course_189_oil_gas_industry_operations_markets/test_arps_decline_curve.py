from arps_decline_curve import ArpsDeclineCurve

def test_arps_decline():
    q_exp = ArpsDeclineCurve.exponential_rate(1000.0, 0.1, 5.0)
    assert q_exp < 1000.0
    q_hyp = ArpsDeclineCurve.hyperbolic_rate(1000.0, 0.1, 0.5, 5.0)
    assert q_hyp > q_exp
