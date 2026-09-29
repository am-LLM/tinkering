from ecommerce_pricing_engine import ECommercePricingEngine

def test_pricing():
    res = ECommercePricingEngine.calculate_total(100.0, 0.1, 0.08, 5.0)
    assert abs(res["total"] - (90.0 + 7.2 + 5.0)) < 1e-4
