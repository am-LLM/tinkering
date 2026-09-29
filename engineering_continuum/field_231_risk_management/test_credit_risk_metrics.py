from credit_risk_metrics import CreditRiskMetrics

def test_credit_risk():
    el = CreditRiskMetrics.calculate_expected_loss(pd=0.02, lgd=0.45, ead=1000000.0)
    assert abs(el - 9000.0) < 1e-4
