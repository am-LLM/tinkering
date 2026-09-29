from bank_run_fire_sale_spiral import BankLiquidityStressModel

def test_bank_run_stress():
    bank = BankLiquidityStressModel(liquid_reserves=100.0, illiquid_assets=500.0, haircut_fire_sale=0.40)
    res1 = bank.process_withdrawal_shock(80.0)
    assert res1["insolvent"] is False
    assert bank.reserves == 20.0
    res2 = bank.process_withdrawal_shock(200.0)
    assert res2["insolvent"] is False
    assert bank.assets < 500.0
