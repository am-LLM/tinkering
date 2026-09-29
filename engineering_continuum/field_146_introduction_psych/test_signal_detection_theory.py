from signal_detection_theory import SignalDetectionTheory

def test_sdt():
    res = SignalDetectionTheory.calculate_sdt(0.84, 0.16)
    assert abs(res["d_prime"] - 2.0) < 0.1
    assert abs(res["criterion_c"]) < 0.1
