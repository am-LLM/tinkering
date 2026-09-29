from csp_motor_imagery_bci import CSPMotorImageryBCI
import numpy as np

def test_cspmotorimagerybci_response():
    engine = CSPMotorImageryBCI(nominal_scale=2.0, channels=4)
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    resp = engine.compute_response(sig)
    assert len(resp) == 4
    assert np.all(np.isfinite(resp))
    assert engine.energy_metric() > 0.0

def test_cspmotorimagerybci_step():
    engine = CSPMotorImageryBCI(nominal_scale=1.5, channels=3)
    val = engine.step_simulation(dt=0.05)
    assert np.isfinite(val)
