from one_time_pad_stream_cipher import OneTimePad
def test_otp_encryption_decryption():
    msg = b"CONFIDENTIAL ASSISTIVE PROTOCOL"
    key = OneTimePad.generate_key(len(msg))
    ciphertext = OneTimePad.xor_bytes(msg, key)
    decrypted = OneTimePad.xor_bytes(ciphertext, key)
    assert decrypted == msg
