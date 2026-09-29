from prm_motion_planner import PRMMotionPlanner

def test_prm():
    prm = PRMMotionPlanner()
    prm.add_roadmap_edge((0, 0), (1, 0))
    prm.add_roadmap_edge((1, 0), (1, 1))
    cost = prm.query((0, 0), (1, 1))
    assert cost == 2.0
