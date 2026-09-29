from nist_risk_engine import NISTRiskEngine

def test_nist():
    e = NISTRiskEngine()
    e.add_control("Protect", "Auth", 1.0, 0.9)
    assert e.domain_maturity("Protect") == 0.9
