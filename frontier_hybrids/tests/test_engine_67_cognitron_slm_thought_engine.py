"""
Unit Tests for Engine 67: COGNITRON-1.58b SLM Thought Engine
"""

import numpy as np
import pytest
from engine_67_cognitron_slm_thought_engine import (
    BitNetTernaryLayer,
    SparseAutoencoderCognitiveFirewall,
    CognitronThoughtEngine
)

def test_bitnet_ternary_layer_quantization():
    layer = BitNetTernaryLayer(in_features=16, out_features=16)
    
    # Verify weights are strictly ternary {-1, 0, 1}
    unique_weights = np.unique(layer.W_ternary)
    for w in unique_weights:
        assert w in [-1, 0, 1]
        
    x = np.random.randn(16)
    out = layer.forward(x)
    assert out.shape == (16,)
    assert not np.any(np.isnan(out))

def test_sae_mechanistic_activation_clamping():
    sae = SparseAutoencoderCognitiveFirewall(d_model=16, n_features=64)
    
    # Input clean embedding
    clean_x = np.random.randn(16)
    _, threat_detected = sae.filter_and_decode(clean_x)
    
    # Inject adversarial prompt injection representation directly into clamped feature direction
    adv_x = np.zeros(16)
    adv_x += sae.W_enc[:, 3] * 10.0  # Force feature 3 to fire strongly
    
    recon, threat_detected = sae.filter_and_decode(adv_x)
    assert threat_detected

def test_cognitron_system2_mcts_reasoning_pipeline():
    engine = CognitronThoughtEngine(d_model=16)
    
    prompt_emb = np.random.randn(16) * 0.5
    result = engine.think(prompt_emb, prompt_text="Verify Invariant Security Protocol")
    
    assert result["test_time_iterations_scaled"] >= 5
    assert len(result["reasoning_trajectory"]) > 0
    assert np.all(np.abs(result["converged_thought_state"]) <= 3.0)
    
    # Verify each node in reasoning trajectory satisfies formal safety
    for step in result["reasoning_trajectory"]:
        assert step["formally_verified"]
