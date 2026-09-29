from buck_state_space_averaged import BuckConverterStateSpace
def test_buck_output_voltage():
    buck = BuckConverterStateSpace(v_in=12.0)
    # Duty cycle 0.5 -> Expected steady state output = 6.0V
    for _ in range(10000):
        v_out = buck.step(duty_cycle=0.5)
    assert 5.5 < v_out < 6.5
