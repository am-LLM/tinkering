from child_resilience_assessor import ChildResilienceAssessor

def test_resilience():
    res = ChildResilienceAssessor.calculate_resilience_index(1, 4)
    assert res["resilience_ratio"] == 2.0
    assert res["tier"] == "HIGH_RESILIENCE"
