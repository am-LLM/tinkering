"""Course 035: Perfect Secrecy One-Time Pad & Stream Cipher Engine"""
import os

class OneTimePad:
    @staticmethod
    def generate_key(length: int) -> bytes:
        return os.urandom(length)

    @staticmethod
    def xor_bytes(data: bytes, key: bytes) -> bytes:
        if len(data) != len(key):
            raise ValueError("Key must match data length for OTP")
        return bytes([d ^ k for d, k in zip(data, key)])
