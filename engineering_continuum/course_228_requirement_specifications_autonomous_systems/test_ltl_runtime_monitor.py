from ltl_runtime_monitor import LTLRuntimeMonitor

def test_ltl_monitor():
    trace = [1, 2, 3, 4, 5]
    assert LTLRuntimeMonitor.check_globally_p(trace, lambda x: x > 0) is True
    assert LTLRuntimeMonitor.check_finally_p(trace, lambda x: x == 3) is True
