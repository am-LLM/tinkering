from six_sigma_cpk_engine import SixSigmaCpkEngine

def test_cpk():
    res = SixSigmaCpkEngine.calculate_cp_cpk(usl=16.0, lsl=4.0, mean=10.0, std=2.0)
    assert res["cp"] == 1.0
    assert res["cpk"] == 1.0
