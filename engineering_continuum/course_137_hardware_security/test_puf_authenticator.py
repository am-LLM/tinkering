from puf_authenticator import PUFAuthenticator

def test_puf():
    puf = PUFAuthenticator(b"hw_seed_123")
    r1 = puf.generate_response(b"chal1")
    r2 = puf.generate_response(b"chal1")
    assert PUFAuthenticator.hamming_distance(r1, r2) == 0
