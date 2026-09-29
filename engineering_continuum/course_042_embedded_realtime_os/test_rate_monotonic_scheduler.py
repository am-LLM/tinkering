from rate_monotonic_scheduler import RateMonotonicScheduler

def test_rms_schedulability():
    tasks = [
        {"id": "T1", "c": 1, "t": 4},
        {"id": "T2", "c": 2, "t": 6},
        {"id": "T3", "c": 3, "t": 12}
    ]
    rms = RateMonotonicScheduler(tasks)
    assert rms.utilization() <= 1.0
    rta = rms.response_time_analysis()
    assert rta["T1"] == 1
    assert rta["T2"] == 3
    assert rta["T3"] == 10
