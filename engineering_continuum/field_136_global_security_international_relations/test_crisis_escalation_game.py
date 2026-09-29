from crisis_escalation_game import CrisisEscalationGame

def test_crisis_game():
    out = CrisisEscalationGame.subgame_perfect_outcome(10.0, 15.0)
    assert out == "CHALLENGER_BACKS_DOWN"
