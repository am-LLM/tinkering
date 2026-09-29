from autocatalytic_network_sim import AutocatalyticNetworkSim

def test_autocatalytic():
    sim = AutocatalyticNetworkSim(0.5, 0.05)
    x = sim.step_replicator(1.0, 1.0, 0.1)
    assert x > 1.0
