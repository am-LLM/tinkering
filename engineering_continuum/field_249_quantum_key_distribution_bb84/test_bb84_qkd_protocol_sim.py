from bb84_qkd_protocol_sim import BB84QKDProtocol
import numpy as np

def test_bb84_noiseless_channel():
    np.random.seed(42)
    qkd = BB84QKDProtocol(n_bits=500, eve_present=False, channel_noise_error_rate=0.0)
    res = qkd.run_protocol()
    assert res["qber"] == 0.0
    assert res["secure"] is True

def test_bb84_eve_detection():
    np.random.seed(42)
    qkd = BB84QKDProtocol(n_bits=2000, eve_present=True, channel_noise_error_rate=0.0)
    res = qkd.run_protocol()
    # Eve interception adds ~25% QBER on sifted bits
    assert res["qber"] > 0.15
    assert res["secure"] is False
