from cdo_tranche_waterfall import CDOTrancheWaterfall

def test_cdo_losses():
    cdo = CDOTrancheWaterfall(0.05, 0.15, 0.80)
    res = cdo.evaluate_losses(0.10) # 10% total loss
    assert res["equity_loss_pct"] == 1.0 # Equity wiped
    assert res["mezzanine_loss_pct"] == 0.3333
    assert res["senior_loss_pct"] == 0.0
