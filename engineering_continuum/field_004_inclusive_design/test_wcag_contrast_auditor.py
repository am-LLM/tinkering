from wcag_contrast_auditor import ContrastAuditor
def test_wcag_aaa_contrast():
    black, white = (0, 0, 0), (255, 255, 255)
    ratio = ContrastAuditor.contrast_ratio(black, white)
    assert ratio >= 7.0, f"Expected AAA ratio >= 7.0, got {ratio}"
