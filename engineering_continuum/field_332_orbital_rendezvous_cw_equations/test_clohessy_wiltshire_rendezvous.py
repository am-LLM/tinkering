from clohessy_wiltshire_rendezvous import ClohessyWiltshirePropagator
import numpy as np

def test_cw_relative_motion():
    cw = ClohessyWiltshirePropagator(mean_motion_n=0.001)
    # Target at origin, chaser 100m behind along-track
    state_0 = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    state_t = cw.propagate(state_0, dt=100.0)
    assert len(state_t) == 6
    assert np.all(np.isfinite(state_t))
