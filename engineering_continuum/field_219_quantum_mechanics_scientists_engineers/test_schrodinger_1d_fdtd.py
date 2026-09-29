import numpy as np
from schrodinger_1d_fdtd import Schrodinger1DFDTD

def test_schrodinger_norm():
    sim = Schrodinger1DFDTD()
    psi0 = np.exp(-0.5 * (np.linspace(-5, 5, 100))**2).astype(complex)
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * 0.1)
    psi1 = sim.step(psi0)
    norm = np.sum(np.abs(psi1)**2) * 0.1
    assert abs(norm - 1.0) < 1e-4
