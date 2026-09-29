from did_w3c_credential_suite import DIDW3CCredentialSuite
import numpy as np

def test_didw3ccredentialsuite_response():
    engine = DIDW3CCredentialSuite(nominal_scale=2.0, channels=4)
    sig = np.array([0.5, 1.0, -0.5, -1.0])
    resp = engine.compute_response(sig)
    assert len(resp) == 4
    assert np.all(np.isfinite(resp))
    assert engine.energy_metric() > 0.0

def test_didw3ccredentialsuite_step():
    engine = DIDW3CCredentialSuite(nominal_scale=1.5, channels=3)
    val = engine.step_simulation(dt=0.05)
    assert np.isfinite(val)
