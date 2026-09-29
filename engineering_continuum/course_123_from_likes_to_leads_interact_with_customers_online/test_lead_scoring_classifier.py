from lead_scoring_classifier import LeadScoringClassifier

def test_lead_scoring():
    scorer = LeadScoringClassifier({"page_views": 0.5, "form_submits": 2.0})
    tier = scorer.assign_tier({"page_views": 10, "form_submits": 2})
    assert tier == "MQL_HOT"
