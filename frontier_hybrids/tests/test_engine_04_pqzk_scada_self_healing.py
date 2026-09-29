from engine_04_pqzk_scada_self_healing import PQZKSCADASelfHealingEngine

def test_pqzk_scada():
    engine = PQZKSCADASelfHealingEngine(num_substations=6, seed=42)
    # Inject Modbus MITM attack
    engine.inject_modbus_tampering(2, 45.0)
    res = engine.self_heal_network()
    assert res["isolated_nodes"] >= 1
    assert engine.nodes[2].breaker_status is False
