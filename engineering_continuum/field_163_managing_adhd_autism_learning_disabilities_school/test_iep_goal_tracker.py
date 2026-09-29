from iep_goal_tracker import IEPGoalTracker

def test_iep():
    iep = IEPGoalTracker("Alex")
    iep.add_goal("G1", "Reading fluency", 80.0)
    iep.record_progress("G1", 85.0)
    iep.record_progress("G1", 90.0)
    iep.record_progress("G1", 82.0)
    assert iep.is_mastered("G1") is True
