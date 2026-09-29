from clinical_diagnostic_metrics import ClinicalDiagnosticMetrics

def test_clinical_metrics():
    m = ClinicalDiagnosticMetrics.compute_metrics(tp=80, fp=10, tn=90, fn=20)
    assert abs(m["sensitivity"] - 0.8) < 1e-5
    assert abs(m["specificity"] - 0.9) < 1e-5
    assert abs(m["youden_j"] - 0.7) < 1e-5
