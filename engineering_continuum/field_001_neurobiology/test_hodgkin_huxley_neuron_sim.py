from hodgkin_huxley_neuron_sim import HodgkinHuxleyNeuron
def test_hh_action_potential():
    neuron = HodgkinHuxleyNeuron()
    v_trace = [neuron.step(i_inj=15.0) for _ in range(500)]
    assert max(v_trace) > 0.0, "Expected action potential spike exceeding 0 mV"
