"""Course 137: Physical Unclonable Function (PUF) Challenge-Response Authenticator"""
import hashlib

class PUFAuthenticator:
    def __init__(self, hardware_seed: bytes):
        self.seed = hardware_seed

    def generate_response(self, challenge: bytes) -> bytes:
        return hashlib.sha256(self.seed + challenge).digest()

    @staticmethod
    def hamming_distance(r1: bytes, r2: bytes) -> int:
        return sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(r1, r2))
