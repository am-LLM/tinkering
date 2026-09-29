from clv_cohort_analyzer import CLVCohortAnalyzer

def test_clv_metrics():
    clv = CLVCohortAnalyzer.calculate_clv(50.0, 4.0, 0.6, 0.2)
    assert clv == 600.0
    decay = CLVCohortAnalyzer.cohort_retention_decay(1000, [1.0, 0.5, 0.25])
    assert decay == [1000, 500, 250]
