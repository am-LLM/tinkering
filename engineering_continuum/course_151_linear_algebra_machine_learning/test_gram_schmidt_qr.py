import numpy as np
from gram_schmidt_qr import GramSchmidtQR

def test_qr():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    q, r = GramSchmidtQR.qr_decompose(a)
    assert np.allclose(q @ r, a)
    assert np.allclose(q.T @ q, np.eye(2))
