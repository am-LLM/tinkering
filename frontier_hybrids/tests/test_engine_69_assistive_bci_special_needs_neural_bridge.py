import pytest
import numpy as np
from frontier_hybrids.engine_69_assistive_bci_special_needs_neural_bridge import AssistiveBCINeuralBridgeEngine

def test_engine_69_initialization():
    engine = AssistiveBCINeuralBridgeEngine(sample_rate=250.0, num_channels=8)
    assert engine.sample_rate == 250.0
    assert engine.num_channels == 8
    assert len(engine.target_frequencies) == 4

def test_engine_69_reference_generation():
    engine = AssistiveBCINeuralBridgeEngine()
    refs = engine.generate_reference_signals(10.0, num_samples=500)
    assert refs.shape == (2 * engine.num_harmonics, 500)

def test_engine_69_ssvep_decoding():
    engine = AssistiveBCINeuralBridgeEngine(sample_rate=250.0, num_channels=8)
    
    # Synthesize 12 Hz SSVEP pattern into multi-channel EEG
    t = np.arange(500) / 250.0
    signal_12hz = np.sin(2 * np.pi * 12.0 * t)
    noise = np.random.normal(0, 0.2, (8, 500))
    eeg_synthetic = np.tile(signal_12hz, (8, 1)) + noise
    
    res = engine.decode_neural_intent(eeg_synthetic)
    assert res["decoded_frequency_hz"] == 12.0
    assert res["canonical_correlation"] > 0.5
    assert res["synthesized_intent"] == "ASSISTANCE NEEDED / PAIN"
    assert res["intent_category"] == "HEALTH_ALERT"

def test_engine_69_p300_erp_detection():
    engine = AssistiveBCINeuralBridgeEngine()
    
    # Synthetic epoch with clear P300 deflection at index 130
    epoch = np.random.normal(0, 0.1, (8, 250))
    epoch[:, 125:145] += 2.5 # Positive peak
    
    detected, snr = engine.decode_p300_erp(epoch, stimulus_onset_idx=50)
    assert detected is True
    assert snr > 1.75
