from thevenin_battery_rc import TheveninDualRC

def test_thevenin_battery():
    bat = TheveninDualRC()
    v = bat.step(current_i=10.0, soc=0.8, dt=0.1)
    assert 3.0 < v < 4.2
