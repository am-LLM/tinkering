from wastewater_bod_kinetics import WastewaterBODKinetics

def test_bod_kinetics():
    bod5 = WastewaterBODKinetics.bod_t(bod_ultimate=300.0, k_decay_day=0.23, t_days=5.0)
    assert 200.0 < bod5 < 300.0
