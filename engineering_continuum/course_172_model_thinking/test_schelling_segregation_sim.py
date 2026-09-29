from schelling_segregation_sim import SchellingSegregationSim

def test_schelling():
    sim = SchellingSegregationSim(10, 0.3)
    sat = sim.satisfaction_rate()
    assert 0.0 <= sat <= 1.0
