from aes_ctr_cipher import AESCTRStreamCipher

def test_aes_ctr_symmetric():
    cipher = AESCTRStreamCipher(key=b"key123")
    ct = cipher.process(b"secret", b"nonce888")
    assert cipher.process(ct, b"nonce888") == b"secret"
