from udl_accessibility_evaluator import UDLAccessibilityEvaluator

def test_udl():
    res = UDLAccessibilityEvaluator.audit_alt_text('<img src="test.jpg" alt="test">')
    assert res["ratio"] == 1.0
