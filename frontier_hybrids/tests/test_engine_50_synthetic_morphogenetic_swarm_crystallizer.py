from engine_50_synthetic_morphogenetic_swarm_crystallizer import MorphogeneticSwarmCrystallizerEngine

def test_morphogenetic_swarm():
    engine = MorphogeneticSwarmCrystallizerEngine(num_robots=16, grid_size=12, seed=42)
    for _ in range(15):
        res = engine.step_morphogenesis_and_crystallization(dt=0.1)
    assert 0.0 <= res["crystallization_fraction"] <= 1.0
    assert res["morphogen_activator_entropy"] >= 0.0
