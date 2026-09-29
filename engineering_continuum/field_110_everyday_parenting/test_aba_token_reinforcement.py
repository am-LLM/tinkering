from aba_token_reinforcement import ABATokenReinforcement

def test_token_economy():
    aba = ABATokenReinforcement(target_tokens=3)
    assert aba.reinforce(2) is False
    assert aba.reinforce(1) is True
    assert aba.redeem() is True
    assert aba.redeem() is False
