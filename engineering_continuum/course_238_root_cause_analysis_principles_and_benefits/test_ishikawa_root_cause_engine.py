from ishikawa_root_cause_engine import IshikawaRootCauseEngine

def test_ishikawa():
    eng = IshikawaRootCauseEngine("Motor Overheat")
    eng.add_cause("Machine", "Bearing friction")
    eng.add_cause("Environment", "High ambient temp")
    assert eng.total_causes_logged() == 2
