from engine_23_bacterial_quorum_traffic_arbitration import BacterialQuorumTrafficEngine

def test_quorum_traffic():
    engine = BacterialQuorumTrafficEngine(num_vehicles=8, seed=42)
    for _ in range(15):
        res = engine.step_arbitration(dt=0.5)
    assert res["cleared_vehicles"] > 0
