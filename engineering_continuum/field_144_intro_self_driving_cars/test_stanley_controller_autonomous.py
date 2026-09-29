from stanley_controller_autonomous import StanleyController

def test_stanley():
    ctrl = StanleyController()
    delta = ctrl.compute_steering_angle(0.1, 0.5, 10.0)
    assert abs(delta) > 0.1
