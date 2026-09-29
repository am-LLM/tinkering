from mohr_circle_stress_transform import MohrCircleStress

def test_mohr_circle():
    res = MohrCircleStress.principal_stresses(sigma_x=80.0, sigma_y=-40.0, tau_xy=30.0)
    assert round(res["sigma_1"], 1) == 87.1
    assert round(res["sigma_2"], 1) == -47.1
    assert round(res["tau_max"], 1) == 67.1
