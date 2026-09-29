from critical_path_scheduler import CriticalPathScheduler

def test_critical_path():
    cpm = CriticalPathScheduler()
    cpm.add_activity("Foundations", 5)
    cpm.add_activity("Framing", 7, ["Foundations"])
    cpm.add_activity("Roofing", 4, ["Framing"])
    cpm.add_activity("Plumbing", 3, ["Foundations"])
    res = cpm.compute_schedule()
    assert res["project_duration"] == 16.0
    assert "Roofing" in res["critical_path"]
