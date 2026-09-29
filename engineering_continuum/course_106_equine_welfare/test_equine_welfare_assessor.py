from equine_welfare_assessor import EquineWelfareAssessor

def test_equine():
    bcs = EquineWelfareAssessor.calculate_bcs(5.0, 6.0, 5.0)
    assert bcs == 5.3
    asym = EquineWelfareAssessor.gait_asymmetry_index(0.5, 0.5)
    assert asym == 0.0
