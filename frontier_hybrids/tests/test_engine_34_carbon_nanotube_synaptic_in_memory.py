import numpy as np
from engine_34_carbon_nanotube_synaptic_in_memory import CNTFETSpikingInMemConvEngine

def test_cntfet_conv():
    engine = CNTFETSpikingInMemConvEngine(seed=42)
    patch = np.ones((3, 3)) * 2.0
    res = engine.convolve_spiking_patch(patch, spike_threshold=0.01)
    assert "mac_current_ua" in res
    assert res["output_spike"] in [0.0, 1.0]
