from special_relativity_lorentz import LorentzTransform

def test_lorentz():
    v = 0.6 * LorentzTransform.C
    assert abs(LorentzTransform.gamma(v) - 1.25) < 1e-4
