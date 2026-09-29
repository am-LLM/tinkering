from clt_chebyshev_bounder import CLTChebyshevBounder

def test_chebyshev():
    b = CLTChebyshevBounder.chebyshev_upper_bound(variance=1.0, epsilon=2.0)
    assert b == 0.25
