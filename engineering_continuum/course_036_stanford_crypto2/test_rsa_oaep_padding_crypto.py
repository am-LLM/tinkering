from rsa_oaep_padding_crypto import RSA_OAEP

def test_rsa_oaep_roundtrip():
    oaep = RSA_OAEP(key_bytes=128)
    msg = b"CRITICAL HARDWARE KEY EXCHANGER"
    padded = oaep.pad(msg, label=b"SYSTEM_SEC")
    recovered = oaep.unpad(padded, label=b"SYSTEM_SEC")
    assert recovered == msg
