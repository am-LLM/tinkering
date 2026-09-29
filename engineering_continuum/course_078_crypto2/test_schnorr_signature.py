from schnorr_signature import SchnorrSignature

def test_schnorr_sign():
    s = SchnorrSignature()
    y = s.keygen(123)
    r, sig = s.sign(123, b"data", k=77)
    assert s.verify(y, b"data", r, sig) is True
