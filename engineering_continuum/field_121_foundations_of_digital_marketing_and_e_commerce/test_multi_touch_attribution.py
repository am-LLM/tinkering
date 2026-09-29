from multi_touch_attribution import MultiTouchAttribution

def test_attribution():
    tps = ["FB", "Google", "Email"]
    res = MultiTouchAttribution.linear_attribution(tps, 90.0)
    assert res["FB"] == 30.0
    assert res["Google"] == 30.0
