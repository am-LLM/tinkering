from engine_22_neuromorphic_olfactory_gas_plume import NeuromorphicOlfactoryPlumeEngine

def test_olfactory_uav():
    engine = NeuromorphicOlfactoryPlumeEngine(seed=42)
    for _ in range(25):
        res = engine.step_olfactory_nav(dt=0.2)
    assert res["distance_to_source_m"] >= 0.0
