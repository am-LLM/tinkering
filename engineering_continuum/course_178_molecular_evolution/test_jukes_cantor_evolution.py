from jukes_cantor_evolution import JukesCantorEvolution

def test_jc_dist():
    d = JukesCantorEvolution.jukes_cantor_distance("AAAA", "AAAT")
    assert d > 0.25
