"""Course 036: RSA-OAEP (Optimal Asymmetric Encryption Padding) Engine"""
import hashlib
import os

def mgf1(seed: bytes, length: int, hash_func=hashlib.sha256) -> bytes:
    h_len = hash_func().digest_size
    t = b""
    counter = 0
    while len(t) < length:
        c_bytes = counter.to_bytes(4, byteorder="big")
        t += hash_func(seed + c_bytes).digest()
        counter += 1
    return t[:length]

class RSA_OAEP:
    def __init__(self, key_bytes=128):
        self.k = key_bytes
        self.h_len = hashlib.sha256().digest_size

    def pad(self, message: bytes, label: bytes = b"") -> bytes:
        l_hash = hashlib.sha256(label).digest()
        ps_len = self.k - len(message) - 2 * self.h_len - 2
        if ps_len < 0:
            raise ValueError("Message too long for OAEP key size")
        db = l_hash + (b"\x00" * ps_len) + b"\x01" + message
        seed = os.urandom(self.h_len)
        db_mask = mgf1(seed, self.k - self.h_len - 1)
        masked_db = bytes(a ^ b for a, b in zip(db, db_mask))
        seed_mask = mgf1(masked_db, self.h_len)
        masked_seed = bytes(a ^ b for a, b in zip(seed, seed_mask))
        return b"\x00" + masked_seed + masked_db

    def unpad(self, em: bytes, label: bytes = b"") -> bytes:
        if len(em) != self.k or em[0] != 0:
            raise ValueError("Invalid OAEP encoding")
        masked_seed = em[1: 1 + self.h_len]
        masked_db = em[1 + self.h_len:]
        seed_mask = mgf1(masked_db, self.h_len)
        seed = bytes(a ^ b for a, b in zip(masked_seed, seed_mask))
        db_mask = mgf1(seed, self.k - self.h_len - 1)
        db = bytes(a ^ b for a, b in zip(masked_db, db_mask))
        l_hash = hashlib.sha256(label).digest()
        if db[:self.h_len] != l_hash:
            raise ValueError("Hash verification mismatch")
        sep_idx = db.find(b"\x01", self.h_len)
        if sep_idx == -1:
            raise ValueError("Separator byte not found")
        return db[sep_idx + 1:]
