from spdc_hong_ou_mandel import HOMInterferometer
def test_hom_quantum_dip():
    hom = HOMInterferometer()
    p_zero_delay = hom.coincidence_probability(0.0)
    p_far_delay = hom.coincidence_probability(50.0)
    assert p_zero_delay == 0.0, "Expected perfect quantum destructive interference at delay=0"
    assert p_far_delay > 0.4
