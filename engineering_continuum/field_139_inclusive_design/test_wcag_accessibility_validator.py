from wcag_accessibility_validator import WCAGAccessibilityValidator

def test_wcag_contrast():
    cr = WCAGAccessibilityValidator.contrast_ratio((0, 0, 0), (255, 255, 255))
    assert cr == 21.0
