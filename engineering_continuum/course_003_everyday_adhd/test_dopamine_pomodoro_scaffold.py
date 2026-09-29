from dopamine_pomodoro_scaffold import ADHDIntervalScaffold
def test_adhd_reward():
    scaffold = ADHDIntervalScaffold(base_work_min=15, micro_reward_points=10)
    earned = scaffold.record_focus_streak(30)
    assert earned == 20
    assert scaffold.points == 20
