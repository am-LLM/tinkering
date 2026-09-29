from monod_bioreactor_chemostat import MonodBioreactorChemostat

def test_chemostat():
    bio = MonodBioreactorChemostat(mu_max=0.8, ks=0.2)
    s = bio.steady_state_substrate(0.4)
    assert abs(s - 0.2) < 1e-4
