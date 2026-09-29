from gray_zone_hybrid_conflict_game import HybridConflictGame

def test_gray_zone_dynamics():
    game = HybridConflictGame(escalation_threshold=0.85)
    res = game.execute_sub_threshold_action({
        "cyber_disruption": 0.4,
        "info_ops": 0.3,
        "infrastructure_probe": 0.2
    })
    assert res["threshold_breached"] is False
    assert res["attribution"] > 0.0
