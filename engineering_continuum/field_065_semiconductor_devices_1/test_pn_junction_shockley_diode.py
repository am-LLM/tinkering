from pn_junction_shockley_diode import PNJunctionModel

def test_pn_junction():
    pn = PNJunctionModel()
    assert 0.6 < pn.v_bi < 0.85
    i_forward = pn.shockley_current(0.7)
    assert i_forward > 1e-4
