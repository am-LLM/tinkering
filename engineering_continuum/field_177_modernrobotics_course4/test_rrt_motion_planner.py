from rrt_motion_planner import RRTPlanner

def test_rrt():
    planner = RRTPlanner((0, 0), (10, 10))
    pt = planner.step((5, 0))
    assert pt == (1.0, 0.0)
