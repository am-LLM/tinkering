from sei_capacity_fade_kalman import BatterySOHEstimator
def test_soh_degradation():
    soh_est = BatterySOHEstimator(nominal_capacity_ah=2.50)
    soh = [soh_est.step_cycle() for _ in range(100)]
    assert soh[-1] < 100.0
    assert soh[-1] > 80.0
