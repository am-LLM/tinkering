from rapid_pfa_triage import RAPIDPFATriage

def test_pfa_triage():
    res = RAPIDPFATriage.triage_severity(3, 3, 2)
    assert res["triage_score"] == 8
    assert res["priority"] == "HIGH_PRIORITY_INTERVENTION"
