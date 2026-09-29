from plasma_debye_frequency import PlasmaDebyeFrequency

def test_plasma():
    w = PlasmaDebyeFrequency.plasma_frequency_rad_s(1e18)
    assert w > 1e10
