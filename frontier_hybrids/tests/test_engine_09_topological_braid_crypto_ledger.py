from engine_09_topological_braid_crypto_ledger import TopologicalBraidConsensusEngine

def test_topological_braid_ledger():
    engine = TopologicalBraidConsensusEngine(num_strands=4, seed=42)
    res = engine.verify_ledger_block_consensus("TX_HASH_99182")
    assert res["braid_depth"] > 0
    assert "invariant_real" in res
