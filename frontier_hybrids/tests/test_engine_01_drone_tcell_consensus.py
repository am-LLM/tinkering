import numpy as np
from engine_01_drone_tcell_consensus import DroneTCellConsensusEngine

def test_drone_tcell_engine():
    engine = DroneTCellConsensusEngine(num_drones=8, seed=42)
    engine.inject_byzantine_fault(0, np.array([100.0, 100.0, 100.0]))
    for _ in range(15):
        m = engine.step_consensus(dt=0.1)
    assert m["active_drones"] == 7
    assert engine.drones[0].is_byzantine
    assert engine.drones[0].is_suppressed
