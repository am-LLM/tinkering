from frontier_hybrids.engine_80_decoy_swarm_rcs_manipulation import DecoySwarmRCSManipulationEngine

def test_engine_80_initialization():
    engine = DecoySwarmRCSManipulationEngine(num_nodes=4)
    assert len(engine.node_roles) == 4
    assert engine.num_nodes == 4

def test_engine_80_rcs_amplification():
    engine = DecoySwarmRCSManipulationEngine()
    
    physical = 0.05
    # Without augmentation
    base = engine.synthesize_augmented_rcs(physical, False, False)
    assert base == 0.05
    
    # With Luneberg lens + active RF spoofing
    boosted = engine.synthesize_augmented_rcs(physical, True, True)
    assert boosted > 20.0 # Emulates heavy aircraft RCS

def test_engine_80_dynamic_role_rotation():
    engine = DecoySwarmRCSManipulationEngine()
    
    # Batteries: Node 0 is dead (15%), Node 2 and 3 are healthy (90%, 85%)
    batteries = [15.0, 45.0, 90.0, 85.0]
    res = engine.execute_swarm_deception_cycle(
        radar_threat_azimuth_deg=45.0,
        radar_power_received_dbm=-40.0,
        node_battery_pcts=batteries
    )
    
    # Healthiest node (Node 2) must become primary decoy bait
    assert res["primary_bait_node_id"] == 2
    # Depleted node (Node 0) must be protected in non-bait role
    assert res["swarm_roles"][0] in ["SILENT_RECON", "SIGINT_COLLECTOR"]
    assert res["deception_ratio"] > 100.0 # Huge contrast between bait and stealth nodes
