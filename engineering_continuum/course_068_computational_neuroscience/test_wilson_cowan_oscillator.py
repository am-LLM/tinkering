from wilson_cowan_oscillator import WilsonCowanModel

def test_wilson_cowan_dynamics():
    wc = WilsonCowanModel()
    trajectory = [wc.step(p=1.5, dt=0.1) for _ in range(100)]
    assert len(trajectory) == 100
    e_vals, i_vals = zip(*trajectory)
    assert all(0.0 <= e <= 1.0 for e in e_vals)
    assert all(0.0 <= i <= 1.0 for i in i_vals)
