from biquad_iir_filter import BiquadIIRFilter

def test_biquad_step():
    filt = BiquadIIRFilter(b=[0.1, 0.2, 0.1], a=[1.0, -0.5, 0.2])
    out = [filt.step(1.0) for _ in range(50)]
    assert len(out) == 50
    assert abs(out[-1]) < 2.0
