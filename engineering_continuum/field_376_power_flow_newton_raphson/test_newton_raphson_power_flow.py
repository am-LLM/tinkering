from newton_raphson_power_flow import PowerFlowNewtonRaphson
import numpy as np

def test_power_injections_flat_start():
    # 2-bus system with transmission admittance
    y_bus = np.array([[2.0 - 10j, -2.0 + 10j], [-2.0 + 10j, 2.0 - 10j]])
    pf = PowerFlowNewtonRaphson(y_bus)
    v_mag = np.array([1.0, 1.0])
    v_ang = np.array([0.0, 0.0])
    p, q = pf.calculate_power_injections(v_mag, v_ang)
    assert np.allclose(p, 0.0)
    assert np.allclose(q, 0.0)
