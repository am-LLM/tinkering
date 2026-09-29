from projectile_drag_dynamics import ProjectileDragDynamics

def test_projectile():
    dyn = ProjectileDragDynamics()
    x, y, vx, vy = 0.0, 0.0, 10.0, 10.0
    x, y, vx, vy = dyn.step(x, y, vx, vy, dt=0.1)
    assert x > 0.0
    assert vy < 10.0
