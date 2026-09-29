from ltl_timed_automata_monitor import LTLRuntimeMonitor

def test_ltl_safety_monitor():
    # Invariant: Speed <= 50 and Altitude >= 10
    inv1 = lambda s: s.get("speed", 0) <= 50
    inv2 = lambda s: s.get("altitude", 0) >= 10
    
    monitor = LTLRuntimeMonitor([inv1, inv2])
    assert monitor.evaluate_state({"speed": 45, "altitude": 100}) is True
    assert monitor.evaluate_state({"speed": 55, "altitude": 100}) is False
    assert monitor.is_safe() is False
