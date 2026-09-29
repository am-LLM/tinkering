from lanchester_combat_model import LanchesterCombatModel

def test_lanchester():
    b, r = LanchesterCombatModel.square_law_step(100.0, 100.0, 0.1, 0.05, 1.0)
    assert b == 95.0
    assert r == 90.0
