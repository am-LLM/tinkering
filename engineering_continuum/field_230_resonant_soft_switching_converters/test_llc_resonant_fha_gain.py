from llc_resonant_fha_gain import LLCResonantFHAGain

def test_llc_gain():
    gain_res = LLCResonantFHAGain.voltage_gain(1.0) # At resonant frequency
    assert abs(gain_res - 1.0) < 1e-4
