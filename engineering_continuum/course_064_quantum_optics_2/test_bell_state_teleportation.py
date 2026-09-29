import numpy as np
from bell_state_teleportation import QuantumTeleporter

def test_teleportation_fidelity():
    teleporter = QuantumTeleporter()
    fid = teleporter.teleport_state(alpha=0.6, beta=0.8j)
    assert np.isclose(fid, 1.0)
