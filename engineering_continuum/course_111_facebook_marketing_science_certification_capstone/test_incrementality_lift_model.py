from incrementality_lift_model import IncrementalityLiftModel

def test_lift_model():
    res = IncrementalityLiftModel.calculate_lift(150, 1000, 100, 1000)
    assert abs(res["abs_lift"] - 0.05) < 1e-5
    assert abs(res["relative_lift"] - 0.5) < 1e-5
    assert res["z_statistic"] > 2.0
