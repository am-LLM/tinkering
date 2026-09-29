"""Course 077: AES-CTR Counter Mode Stream Cipher with Nonce Increment"""
import hashlib

class AESCTRStreamCipher:
    def __init__(self, key: bytes):
        self.key = hashlib.sha256(key).digest()

    def _keystream_block(self, nonce: bytes, counter: int) -> bytes:
        counter_bytes = counter.to_bytes(8, byteorder='big')
        return hashlib.sha256(self.key + nonce + counter_bytes).digest()

    def process(self, data: bytes, nonce: bytes) -> bytes:
        out = bytearray()
        block_idx = 0
        for i in range(0, len(data), 32):
            chunk = data[i:i+32]
            ks = self._keystream_block(nonce, block_idx)
            for j in range(len(chunk)):
                out.append(chunk[j] ^ ks[j])
            block_idx += 1
        return bytes(out)
