from closed_loop_dbs_controller import ClosedLoopDBSController
import numpy as np

def test_closedloopdbscontroller_response():
    engine = ClosedLoopDBSController(nominal_scale=2.0, channels=4)
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    resp = engine.compute_response(sig)
    assert len(resp) == 4
    assert np.all(np.isfinite(resp))
    assert engine.energy_metric() > 0.0

def test_closedloopdbscontroller_step():
    engine = ClosedLoopDBSController(nominal_scale=1.5, channels=3)
    val = engine.step_simulation(dt=0.05)
    assert np.isfinite(val)
