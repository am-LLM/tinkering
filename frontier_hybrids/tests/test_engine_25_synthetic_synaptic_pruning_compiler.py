from engine_25_synthetic_synaptic_pruning_compiler import SynapticPruningCompilerEngine

def test_synaptic_pruning_compiler():
    engine = SynapticPruningCompilerEngine(seed=42)
    res = engine.simulate_microglial_pruning(activity_threshold=1.0)
    assert res["pruned_instructions"] == 15.0
    assert res["code_size_reduction_pct"] > 0.0
