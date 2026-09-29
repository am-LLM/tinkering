from black_scholes_greeks import BlackScholesGreeks

def test_black_scholes():
    res = BlackScholesGreeks.call_price_and_greeks(s=100.0, k=100.0, t=1.0, r=0.05, sigma=0.2)
    assert res["price"] > 0.0
    assert 0.5 < res["delta"] < 0.7
