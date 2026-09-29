from spc_control_chart_jidoka import SPCProcessMonitor

def test_spc_jidoka():
    spc = SPCProcessMonitor(target_mu=100.0, target_sigma=2.0)
    res_normal = spc.evaluate_sample(101.5)
    assert res_normal["jidoka_andon_stop"] is False
    res_outlier = spc.evaluate_sample(108.5)
    assert res_outlier["jidoka_andon_stop"] is True
