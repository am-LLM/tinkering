import pytest
import numpy as np
from frontier_hybrids.engine_70_interspecies_mammalian_rodent_communication import InterspeciesMammalianRodentCommEngine

def test_engine_70_initialization():
    engine = InterspeciesMammalianRodentCommEngine(sample_rate_hz=250000.0)
    assert engine.sample_rate_hz == 250000.0
    assert "RODENT_50KHZ_PROSOCIAL" in engine.lexicon

def test_engine_70_rodent_50khz_usv_translation():
    engine = InterspeciesMammalianRodentCommEngine(sample_rate_hz=250000.0)
    
    # Synthesize 52 kHz rodent play vocalization
    t = np.arange(2500) / 250000.0 # 10ms
    usv_audio = np.sin(2 * np.pi * 52000.0 * t) + np.random.normal(0, 0.05, len(t))
    
    res = engine.translate_bioacoustic_event(usv_audio)
    assert 48000.0 <= res["peak_frequency_hz"] <= 56000.0
    assert res["classified_call_type"] == "RODENT_50KHZ_PROSOCIAL"
    assert "PLAY" in res["semantic_translation"]
    assert res["affective_valence"] > 0.5

def test_engine_70_rodent_22khz_alarm_call():
    engine = InterspeciesMammalianRodentCommEngine(sample_rate_hz=250000.0)
    
    # Synthesize 22 kHz alarm call
    t = np.arange(2500) / 250000.0
    alarm_audio = np.sin(2 * np.pi * 22000.0 * t) + np.random.normal(0, 0.05, len(t))
    
    res = engine.translate_bioacoustic_event(alarm_audio)
    assert 20000.0 <= res["peak_frequency_hz"] <= 24000.0
    assert res["classified_call_type"] == "RODENT_22KHZ_ALARM"
    assert "ALARM" in res["semantic_translation"]
    assert res["affective_valence"] < 0.0

def test_engine_70_mammalian_low_freq_purr():
    engine = InterspeciesMammalianRodentCommEngine(sample_rate_hz=10000.0)
    
    # Synthesize 100 Hz mammalian purr/trill
    t = np.arange(2000) / 10000.0
    purr_audio = np.sin(2 * np.pi * 100.0 * t)
    
    res = engine.translate_bioacoustic_event(purr_audio)
    assert res["classified_call_type"] in ["MAMMAL_AFFILIATIVE_PURR_TRILL", "MAMMAL_LOW_GROWL_THREAT"]
    assert res["confidence"] > 0.5
