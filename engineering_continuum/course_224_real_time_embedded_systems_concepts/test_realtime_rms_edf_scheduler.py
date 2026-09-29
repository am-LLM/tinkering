from realtime_rms_edf_scheduler import RealTimeSchedulability

def test_realtime_sched():
    assert RealTimeSchedulability.check_rms_schedulable([1, 2], [5, 10]) is True
    assert RealTimeSchedulability.check_edf_schedulable([3, 4], [5, 10]) is True
