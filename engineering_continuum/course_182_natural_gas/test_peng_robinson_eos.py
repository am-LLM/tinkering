from peng_robinson_eos import PengRobinsonEOS

def test_pr_eos():
    tr, pr = PengRobinsonEOS.calculate_reduced_params(300.0, 50.0)
    assert tr > 1.0
    assert pr > 1.0
