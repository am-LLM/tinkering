from q_law_low_thrust_transfer import QLawLowThrustTransfer
import numpy as np

def test_qlawlowthrusttransfer_loop():
    engine = QLawLowThrustTransfer(damping=0.9, state_dim=4)
    u = np.array([1.0, 2.0, 3.0, 4.0])
    st = engine.execute_core_loop(u)
    assert len(st) == 4
    assert engine.check_boundedness()
    assert engine.get_energy() > 0.0

def test_qlawlowthrusttransfer_convergence():
    engine = QLawLowThrustTransfer(damping=0.5, state_dim=3)
    for _ in range(50):
        engine.execute_core_loop(np.zeros(3))
    assert engine.get_energy() < 0.1
