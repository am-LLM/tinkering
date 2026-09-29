from synaptic_quantal_release import SynapticQuantalRelease

def test_synaptic_release():
    m = SynapticQuantalRelease.mean_quantal_content(10, 0.3)
    assert m == 3.0
