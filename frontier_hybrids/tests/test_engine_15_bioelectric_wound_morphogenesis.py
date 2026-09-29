from engine_15_bioelectric_wound_morphogenesis import BioelectricSMAMorphogenesisEngine

def test_bioelectric_sma():
    engine = BioelectricSMAMorphogenesisEngine(grid_size=8, seed=42)
    engine.inflict_geometric_wound(4, 4, radius=2)
    res = engine.step_morphogenesis(dt=0.1)
    assert res["depolarized_wound_nodes"] > 0
    assert res["max_sma_strain"] >= 0.0
