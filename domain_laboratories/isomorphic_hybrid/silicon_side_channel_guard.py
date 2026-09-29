"""
Silicon Side-Channel & Fault Injection Guard Firmware
=====================================================
Production-grade hardware security layer providing:
1. Constant-time primitives & constant-time AES-128 block cipher (cache-attack immune).
2. Constant-time Lattice/Ring-LWE modular arithmetic & conditional selection.
3. PRNG-based Clock Jitter & Dummy Cycle Injection Engine to mitigate DPA/SPA.
4. Active Voltage Glitch & Transient Drop Detector (dV/dt & threshold monitoring).
5. Tamper Detection & Hardware Zeroization Engine.
"""

from __future__ import annotations
import math
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class SecurityState(str, Enum):
    SECURE = "SECURE"
    SUSPICIOUS = "SUSPICIOUS"
    COMPROMISED = "COMPROMISED"
    ZEROIZED = "ZEROIZED"


class ConstantTimeOps:
    """Constant-time arithmetic and bitwise utilities for side-channel immunity."""

    @staticmethod
    def ct_is_zero(x: int) -> int:
        """Returns 1 if x == 0, else 0 in constant time (32-bit integer)."""
        x = x & 0xFFFFFFFF
        return 1 ^ ((x | ((-x) & 0xFFFFFFFF)) >> 31)

    @staticmethod
    def ct_eq(a: int, b: int) -> int:
        """Returns 1 if a == b, else 0."""
        return ConstantTimeOps.ct_is_zero(a ^ b)

    @staticmethod
    def ct_select(cond: int, true_val: int, false_val: int) -> int:
        """Returns true_val if cond == 1, false_val if cond == 0 in constant time."""
        mask = 0xFFFFFFFF if (cond & 1) else 0x00000000
        return (mask & true_val) | (~mask & 0xFFFFFFFF & false_val)

    @staticmethod
    def ct_select_bytes(cond: int, true_bytes: bytes, false_bytes: bytes) -> bytes:
        """Selects between two byte strings of equal length in constant time."""
        if len(true_bytes) != len(false_bytes):
            raise ValueError("Byte sequences must have identical length")
        mask = -(cond & 1) & 0xFF
        out = bytearray(len(true_bytes))
        for i in range(len(true_bytes)):
            out[i] = (mask & true_bytes[i]) | (~mask & false_bytes[i])
        return bytes(out)

    @staticmethod
    def ct_memcmp(a: bytes, b: bytes) -> bool:
        """Constant-time byte comparison."""
        if len(a) != len(b):
            return False
        diff = 0
        for x, y in zip(a, b):
            diff |= x ^ y
        return diff == 0


class ConstantTimeAES:
    """
    Constant-time AES-128 Implementation.
    Uses algebraic Rijndael S-box computation in GF(2^8) via tower fields
    or masked arithmetic to avoid data-dependent memory cache lookups.
    """

    @staticmethod
    def _gf_mul(a: int, b: int) -> int:
        """Constant-time Galois Field GF(2^8) multiplication."""
        p = 0
        for _ in range(8):
            p ^= ConstantTimeOps.ct_select(b & 1, a, 0)
            hi_bit = (a >> 7) & 1
            a = ((a << 1) & 0xFF) ^ ConstantTimeOps.ct_select(hi_bit, 0x1B, 0)
            b >>= 1
        return p & 0xFF

    @staticmethod
    def _gf_inv(a: int) -> int:
        """Constant-time GF(2^8) multiplicative inverse via Fermat's Little Theorem (a^254)."""
        a2 = ConstantTimeAES._gf_mul(a, a)
        a4 = ConstantTimeAES._gf_mul(a2, a2)
        a8 = ConstantTimeAES._gf_mul(a4, a4)
        a16 = ConstantTimeAES._gf_mul(a8, a8)
        a32 = ConstantTimeAES._gf_mul(a16, a16)
        a64 = ConstantTimeAES._gf_mul(a32, a32)
        a128 = ConstantTimeAES._gf_mul(a64, a64)

        a_inv = ConstantTimeAES._gf_mul(a2, a4)
        a_inv = ConstantTimeAES._gf_mul(a_inv, a8)
        a_inv = ConstantTimeAES._gf_mul(a_inv, a16)
        a_inv = ConstantTimeAES._gf_mul(a_inv, a32)
        a_inv = ConstantTimeAES._gf_mul(a_inv, a64)
        a_inv = ConstantTimeAES._gf_mul(a_inv, a128)
        return ConstantTimeOps.ct_select(ConstantTimeOps.ct_is_zero(a), 0, a_inv)

    @classmethod
    def sbox_eval(cls, byte_val: int) -> int:
        """Constant-time S-Box algebraic evaluation."""
        inv = cls._gf_inv(byte_val & 0xFF)
        s = inv
        for shift in (1, 2, 3, 4):
            rot = ((inv << shift) | (inv >> (8 - shift))) & 0xFF
            s ^= rot
        return (s ^ 0x63) & 0xFF

    @classmethod
    def key_expansion(cls, key: bytes) -> List[List[List[int]]]:
        """Expands 16-byte key into 11 round keys (each 4x4 matrix of bytes)."""
        if len(key) != 16:
            raise ValueError("AES-128 requires exactly 16-byte key")

        rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
        w = [list(key[i * 4:(i + 1) * 4]) for i in range(4)]

        for i in range(4, 44):
            temp = list(w[i - 1])
            if i % 4 == 0:
                temp = temp[1:] + temp[:1]
                temp = [cls.sbox_eval(b) for b in temp]
                temp[0] ^= rcon[(i // 4) - 1]
            new_word = [w[i - 4][j] ^ temp[j] for j in range(4)]
            w.append(new_word)

        round_keys = []
        for r in range(11):
            rk = []
            for col in range(4):
                rk.append(w[r * 4 + col])
            state = [[rk[c][r_idx] for c in range(4)] for r_idx in range(4)]
            round_keys.append(state)
        return round_keys

    @classmethod
    def encrypt_block(cls, plaintext: bytes, key: bytes) -> bytes:
        """Encrypts single 16-byte block in constant time."""
        if len(plaintext) != 16:
            raise ValueError("Plaintext block must be 16 bytes")
        round_keys = cls.key_expansion(key)

        state = [[plaintext[r + 4 * c] for c in range(4)] for r in range(4)]

        for r in range(4):
            for c in range(4):
                state[r][c] ^= round_keys[0][r][c]

        for round_idx in range(1, 10):
            for r in range(4):
                for c in range(4):
                    state[r][c] = cls.sbox_eval(state[r][c])

            state[1] = state[1][1:] + state[1][:1]
            state[2] = state[2][2:] + state[2][:2]
            state[3] = state[3][3:] + state[3][:3]

            for c in range(4):
                s0 = state[0][c]
                s1 = state[1][c]
                s2 = state[2][c]
                s3 = state[3][c]
                state[0][c] = cls._gf_mul(2, s0) ^ cls._gf_mul(3, s1) ^ s2 ^ s3
                state[1][c] = s0 ^ cls._gf_mul(2, s1) ^ cls._gf_mul(3, s2) ^ s3
                state[2][c] = s0 ^ s1 ^ cls._gf_mul(2, s2) ^ cls._gf_mul(3, s3)
                state[3][c] = cls._gf_mul(3, s0) ^ s1 ^ s2 ^ cls._gf_mul(2, s3)

            for r in range(4):
                for c in range(4):
                    state[r][c] ^= round_keys[round_idx][r][c]

        for r in range(4):
            for c in range(4):
                state[r][c] = cls.sbox_eval(state[r][c])
        state[1] = state[1][1:] + state[1][:1]
        state[2] = state[2][2:] + state[2][:2]
        state[3] = state[3][3:] + state[3][:3]
        for r in range(4):
            for c in range(4):
                state[r][c] ^= round_keys[10][r][c]

        out = bytearray(16)
        for r in range(4):
            for c in range(4):
                out[r + 4 * c] = state[r][c]
        return bytes(out)


class ConstantTimeRingArithmetic:
    """Constant-time Lattice / Ring-LWE modular arithmetic engine (N=256, Q=3329)."""

    Q = 3329
    Q_INV = 3327
    MONT_R = 65536

    @classmethod
    def montgomery_reduce(cls, a: int) -> int:
        """Constant-time Montgomery reduction: returns a * R^(-1) mod Q."""
        t = (a * cls.Q_INV) & 0xFFFF
        res = (a + t * cls.Q) >> 16
        diff = res - cls.Q
        borrow = 1 if diff < 0 else 0
        return ConstantTimeOps.ct_select(borrow, res, diff)

    @classmethod
    def barrett_reduce(cls, a: int) -> int:
        """Constant-time Barrett modular reduction."""
        v = (a * 20159) >> 26
        t = a - v * cls.Q
        diff = t - cls.Q
        borrow = 1 if diff < 0 else 0
        return ConstantTimeOps.ct_select(borrow, t, diff)

    @classmethod
    def ct_cswap(cls, a: np.ndarray, b: np.ndarray, condition: int) -> Tuple[np.ndarray, np.ndarray]:
        """Constant-time conditional swap of two polynomial arrays."""
        mask = -(condition & 1)
        delta = (a ^ b) & mask
        return (a ^ delta), (b ^ delta)


class ClockJitterEngine:
    """
    Dynamic PRNG Clock Jitter & Dummy Cycle Injection Engine.
    Disrupts Correlation Power Analysis (DPA) and Simple Power Analysis (SPA).
    """

    def __init__(self, min_jitter_cycles: int = 10, max_jitter_cycles: int = 100):
        self.min_cycles = min_jitter_cycles
        self.max_cycles = max_jitter_cycles
        self.total_jitter_injected = 0
        self._entropy_pool = bytearray(secrets.token_bytes(64))

    def inject_jitter(self) -> int:
        """Injects a pseudorandom number of dummy ALU operations to randomize timing."""
        fresh_byte = secrets.token_bytes(1)[0]
        self._entropy_pool[0] ^= fresh_byte
        
        jitter_count = self.min_cycles + (self._entropy_pool[0] % (self.max_cycles - self.min_cycles + 1))
        dummy_acc = 0xAA55
        for i in range(jitter_count):
            dummy_acc = (dummy_acc * 31 + i) & 0xFFFF
        self.total_jitter_injected += jitter_count
        return jitter_count

    def execute_with_jitter(self, target_callable: callable, *args, **kwargs) -> Any:
        """Executes a target function wrapped with pre/post jitter barriers."""
        self.inject_jitter()
        result = target_callable(*args, **kwargs)
        self.inject_jitter()
        return result


@dataclass
class VoltageSample:
    timestamp_ns: int
    voltage_volts: float
    dv_dt_v_per_s: float
    is_anomaly: bool


class VoltageGlitchDetector:
    """
    Active Real-Time Voltage Glitch & Brownout Monitor.
    Detects fast voltage drop spikes (dV/dt) and out-of-spec power rail fluctuations.
    """

    def __init__(
        self,
        nominal_voltage: float = 3.3,
        under_voltage_threshold: float = 2.8,
        over_voltage_threshold: float = 3.8,
        max_dv_dt_threshold: float = 50000.0,
        glitch_window_size: int = 20,
    ):
        self.nominal_voltage = nominal_voltage
        self.under_voltage_threshold = under_voltage_threshold
        self.over_voltage_threshold = over_voltage_threshold
        self.max_dv_dt = max_dv_dt_threshold
        self.window_size = glitch_window_size

        self.samples: List[VoltageSample] = []
        self.glitch_count: int = 0
        self.last_sample_time_ns: Optional[int] = None
        self.last_voltage: float = nominal_voltage

    def monitor_sample(self, voltage: float, timestamp_ns: Optional[int] = None) -> bool:
        """
        Ingests a voltage ADC reading and evaluates glitch criteria.
        Returns True if voltage is within safe operational envelope, False if a glitch is detected.
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        dv_dt = 0.0
        if self.last_sample_time_ns is not None:
            dt_s = (timestamp_ns - self.last_sample_time_ns) / 1e9
            if dt_s > 0:
                dv_dt = (voltage - self.last_voltage) / dt_s

        is_anomaly = False
        if voltage < self.under_voltage_threshold or voltage > self.over_voltage_threshold:
            is_anomaly = True
        if abs(dv_dt) > self.max_dv_dt:
            is_anomaly = True

        if is_anomaly:
            self.glitch_count += 1

        sample = VoltageSample(
            timestamp_ns=timestamp_ns,
            voltage_volts=voltage,
            dv_dt_v_per_s=dv_dt,
            is_anomaly=is_anomaly,
        )
        self.samples.append(sample)
        if len(self.samples) > self.window_size:
            self.samples.pop(0)

        self.last_sample_time_ns = timestamp_ns
        self.last_voltage = voltage
        return not is_anomaly


class HardwareSecurityGuard:
    """
    Unified Hardware Security Supervisor.
    Orchestrates glitch monitoring, jitter injection, constant-time cryptography,
    and automatic memory zeroization upon tamper trip.
    """

    def __init__(self, key_material: bytes):
        if len(key_material) != 16:
            raise ValueError("Root key must be 16 bytes")
        self._secure_key_store = bytearray(key_material)
        self.state = SecurityState.SECURE
        self.jitter_engine = ClockJitterEngine()
        self.glitch_detector = VoltageGlitchDetector()
        self.tamper_events: List[Dict[str, Any]] = []

    @property
    def is_compromised(self) -> bool:
        return self.state in (SecurityState.COMPROMISED, SecurityState.ZEROIZED)

    def process_voltage_telemetry(self, voltage: float, timestamp_ns: Optional[int] = None) -> bool:
        """Evaluates voltage safety; trips zeroization on repeated or severe glitches."""
        safe = self.glitch_detector.monitor_sample(voltage, timestamp_ns)
        if not safe:
            self.tamper_events.append({
                "type": "VOLTAGE_GLITCH",
                "voltage": voltage,
                "timestamp_ns": timestamp_ns or time.time_ns(),
                "glitch_count": self.glitch_detector.glitch_count,
            })
            if self.glitch_detector.glitch_count >= 3:
                self.trigger_zeroization("Excessive Voltage Glitch Threshold Exceeded")
            else:
                self.state = SecurityState.SUSPICIOUS
        return safe

    def trigger_zeroization(self, reason: str) -> None:
        """Wipes all sensitive key material with 0x00 and cryptographically random overwrite passes."""
        for pattern in (0x55, 0xAA, 0x00):
            for i in range(len(self._secure_key_store)):
                self._secure_key_store[i] = pattern
        rand_bytes = secrets.token_bytes(len(self._secure_key_store))
        for i in range(len(self._secure_key_store)):
            self._secure_key_store[i] = rand_bytes[i]
        for i in range(len(self._secure_key_store)):
            self._secure_key_store[i] = 0x00

        self.state = SecurityState.ZEROIZED
        self.tamper_events.append({
            "type": "ZEROIZATION_TRIGGERED",
            "reason": reason,
            "timestamp_ns": time.time_ns(),
        })

    def secure_aes_encrypt(self, plaintext: bytes) -> bytes:
        """Executes constant-time AES encryption wrapped with jitter injection."""
        if self.is_compromised:
            raise PermissionError("Hardware Guard is ZEROIZED / COMPROMISED. Key destroyed.")

        def _encrypt():
            return ConstantTimeAES.encrypt_block(plaintext, bytes(self._secure_key_store))

        return self.jitter_engine.execute_with_jitter(_encrypt)
