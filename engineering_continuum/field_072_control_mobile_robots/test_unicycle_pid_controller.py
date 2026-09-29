from unicycle_pid_controller import UnicycleController
import math

def test_unicycle_convergence():
    ctrl = UnicycleController()
    x, y, th = 0.0, 0.0, 0.0
    target_x, target_y = 5.0, 5.0
    for _ in range(200):
        v, w = ctrl.compute_control(x, y, th, target_x, target_y)
        x, y, th = ctrl.step_kinematics(x, y, th, v, w, dt=0.05)
    assert math.hypot(target_x - x, target_y - y) < 0.2
