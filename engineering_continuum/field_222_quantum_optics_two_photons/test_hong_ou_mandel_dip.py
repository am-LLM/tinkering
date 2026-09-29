from hong_ou_mandel_dip import HongOuMandelDip

def test_hom_dip():
    v = HongOuMandelDip.hom_visibility(10.0, 100.0)
    assert v == 0.90
    p0 = HongOuMandelDip.gaussian_dip_profile(0.0, visibility=1.0)
    assert p0 == 0.0 # Perfect null coincidence at zero delay
