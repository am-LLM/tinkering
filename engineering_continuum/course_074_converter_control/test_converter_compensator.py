from converter_compensator import TypeIICompensator

def test_compensator_regulation():
    comp = TypeIICompensator(v_ref=12.0)
    duty = comp.compute_duty(v_measured=10.0, dt=1e-3)
    assert 0.0 <= duty <= 1.0
