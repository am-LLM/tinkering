from adaptive_behavior_scorer import AdaptiveBehaviorScorer

def test_adaptive_scorer():
    res = AdaptiveBehaviorScorer.composite_score(85, 80, 75)
    assert res["general_adaptive_composite"] == 80.0
    assert res["support_tier"] == "Borderline"
