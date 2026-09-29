from skid_steer_kinematics import SkidSteerKinematics

def test_skid_steer():
    bot = SkidSteerKinematics(track_width_m=0.5, wheel_radius_m=0.1)
    v, w = bot.forward_kinematics(10.0, 10.0)
    assert v == 1.0
    assert w == 0.0
