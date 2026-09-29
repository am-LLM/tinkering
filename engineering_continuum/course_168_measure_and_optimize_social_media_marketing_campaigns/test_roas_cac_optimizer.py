from roas_cac_optimizer import ROASCACOptimizer

def test_roas():
    res = ROASCACOptimizer.evaluate_campaign(1000.0, 4000.0, 50)
    assert res["roas"] == 4.0
    assert res["cac"] == 20.0
