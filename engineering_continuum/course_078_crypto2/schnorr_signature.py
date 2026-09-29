"""Course 078: Schnorr Digital Signature Scheme over Prime Subgroup"""
import hashlib

class SchnorrSignature:
    def __init__(self, p: int = 1000000007, q: int = 500000003, g: int = 2):
        self.p, self.q, self.g = p, q, g

    def keygen(self, secret_x: int) -> int:
        return pow(self.g, secret_x, self.p)

    def sign(self, secret_x: int, message: bytes, k: int) -> tuple:
        r = pow(self.g, k, self.p)
        h = int(hashlib.sha256(r.to_bytes(32, 'big') + message).hexdigest(), 16) % self.q
        s = (k - secret_x * h) % self.q
        return r, s

    def verify(self, public_y: int, message: bytes, r: int, s: int) -> bool:
        h = int(hashlib.sha256(r.to_bytes(32, 'big') + message).hexdigest(), 16) % self.q
        v = (pow(self.g, s, self.p) * pow(public_y, h, self.p)) % self.p
        return v == (r % self.p)
