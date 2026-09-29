"""
Radiation-Hardened Firmware & SEU Fault Mitigation Layer
========================================================
Implements:
1. Automated SEC-DED (Single Error Correction, Double Error Detection) Hamming Code Memory Scrubber.
2. Triple-Modular Redundancy (TMR) Attitude Quaternion Voting Engine with Automatic Self-Healing.
3. Fast-Response Overcurrent Crowbar Protection & Thermal Trip Controller.
4. SEU Bit-Flip Fault Injection & Telemetry Diagnostics.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class ECCStatus(str, Enum):
    NO_ERROR = "NO_ERROR"
    SINGLE_ERROR_CORRECTED = "SINGLE_ERROR_CORRECTED"
    DOUBLE_ERROR_DETECTED = "DOUBLE_ERROR_DETECTED"
    UNCORRECTABLE_MULTI_BIT = "UNCORRECTABLE_MULTI_BIT"


class HammingSECDED:
    """
    Extended Hamming (8,4) SEC-DED Codec.
    Encodes 4 bits of data into 8-bit codeword (4 data bits, 3 parity bits, 1 overall parity bit).
    Detects up to 2 bit errors, corrects 1 bit error deterministically.
    """

    # Parity check bit positions (1-indexed for standard Hamming (7,4)):
    # Pos 1: Parity p1 (covers 1, 3, 5, 7) -> bits: p1, d1, d2, d4
    # Pos 2: Parity p2 (covers 2, 3, 6, 7) -> bits: p2, d1, d3, d4
    # Pos 4: Parity p3 (covers 4, 5, 6, 7) -> bits: p3, d2, d3, d4
    # Pos 8: Parity p0 (overall parity of bits 1..7)

    @staticmethod
    def encode_nibble(nibble: int) -> int:
        """Encodes a 4-bit integer (0..15) into an 8-bit SEC-DED codeword."""
        d1 = (nibble >> 0) & 1
        d2 = (nibble >> 1) & 1
        d3 = (nibble >> 2) & 1
        d4 = (nibble >> 3) & 1

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4

        # 7-bit codeword bits: [p1, p2, d1, p3, d2, d3, d4] (indices 1..7)
        c7 = (p1 << 0) | (p2 << 1) | (d1 << 2) | (p3 << 3) | (d2 << 4) | (d3 << 5) | (d4 << 6)
        
        # 8th bit: Overall parity p0
        p0 = bin(c7).count("1") % 2
        return c7 | (p0 << 7)

    @staticmethod
    def decode_nibble(codeword: int) -> Tuple[int, ECCStatus]:
        """
        Decodes an 8-bit SEC-DED codeword into a 4-bit nibble.
        Detects and corrects single-bit errors; detects double-bit errors.
        """
        codeword &= 0xFF
        c7 = codeword & 0x7F
        p0_received = (codeword >> 7) & 1
        p0_computed = bin(c7).count("1") % 2

        overall_parity_error = (p0_received != p0_computed)

        # Extract bits 1..7
        b1 = (c7 >> 0) & 1
        b2 = (c7 >> 1) & 1
        b3 = (c7 >> 2) & 1
        b4 = (c7 >> 3) & 1
        b5 = (c7 >> 4) & 1
        b6 = (c7 >> 5) & 1
        b7 = (c7 >> 6) & 1

        # Syndrome calculation
        s1 = b1 ^ b3 ^ b5 ^ b7
        s2 = b2 ^ b3 ^ b6 ^ b7
        s3 = b4 ^ b5 ^ b6 ^ b7
        syndrome = (s3 << 2) | (s2 << 1) | s1

        if syndrome == 0:
            if overall_parity_error:
                # Error was in the overall parity bit itself
                d1 = b3
                d2 = b5
                d3 = b6
                d4 = b7
                return ((d4 << 3) | (d3 << 2) | (d2 << 1) | d1), ECCStatus.SINGLE_ERROR_CORRECTED
            else:
                d1 = b3
                d2 = b5
                d3 = b6
                d4 = b7
                return ((d4 << 3) | (d3 << 2) | (d2 << 1) | d1), ECCStatus.NO_ERROR
        else:
            if overall_parity_error:
                # Single-bit error in codeword at 1-indexed position = syndrome
                c7_corrected = c7 ^ (1 << (syndrome - 1))
                d1 = (c7_corrected >> 2) & 1
                d2 = (c7_corrected >> 4) & 1
                d3 = (c7_corrected >> 5) & 1
                d4 = (c7_corrected >> 6) & 1
                return ((d4 << 3) | (d3 << 2) | (d2 << 1) | d1), ECCStatus.SINGLE_ERROR_CORRECTED
            else:
                # Double-bit error detected (syndrome != 0 but overall parity matches)
                d1 = b3
                d2 = b5
                d3 = b6
                d4 = b7
                return ((d4 << 3) | (d3 << 2) | (d2 << 1) | d1), ECCStatus.DOUBLE_ERROR_DETECTED

    @classmethod
    def encode_bytes(cls, data: bytes) -> bytes:
        """Encodes raw byte buffer into SEC-DED protected buffer (2 bytes output per 1 byte input)."""
        out = bytearray(len(data) * 2)
        for i, b in enumerate(data):
            lo_nibble = b & 0x0F
            hi_nibble = (b >> 4) & 0x0F
            out[2 * i] = cls.encode_nibble(lo_nibble)
            out[2 * i + 1] = cls.encode_nibble(hi_nibble)
        return bytes(out)

    @classmethod
    def decode_and_correct_bytes(cls, ecc_data: bytes) -> Tuple[bytes, ECCStatus, int]:
        """
        Decodes SEC-DED buffer.
        Returns: (recovered_bytes, worst_status, total_corrections)
        """
        if len(ecc_data) % 2 != 0:
            raise ValueError("ECC buffer length must be even")

        num_bytes = len(ecc_data) // 2
        recovered = bytearray(num_bytes)
        worst_status = ECCStatus.NO_ERROR
        corrections = 0

        for i in range(num_bytes):
            lo_c = ecc_data[2 * i]
            hi_c = ecc_data[2 * i + 1]

            lo_nibble, st_lo = cls.decode_nibble(lo_c)
            hi_nibble, st_hi = cls.decode_nibble(hi_c)

            recovered[i] = (hi_nibble << 4) | lo_nibble

            for st in (st_lo, st_hi):
                if st == ECCStatus.SINGLE_ERROR_CORRECTED:
                    corrections += 1
                    if worst_status == ECCStatus.NO_ERROR:
                        worst_status = ECCStatus.SINGLE_ERROR_CORRECTED
                elif st == ECCStatus.DOUBLE_ERROR_DETECTED:
                    worst_status = ECCStatus.DOUBLE_ERROR_DETECTED

        return bytes(recovered), worst_status, corrections


class MemoryScrubber:
    """
    Autonomous Periodic Memory Scrubber.
    Walks memory pages, detects soft bit flips, corrects single-bit flips in place,
    and logs telemetry on Radiation Single-Event Upsets (SEU).
    """

    def __init__(self, raw_data: bytes):
        self.raw_data_len = len(raw_data)
        self.ecc_buffer = bytearray(HammingSECDED.encode_bytes(raw_data))
        self.total_single_flips_corrected = 0
        self.total_double_flips_detected = 0
        self.scrub_cycles_completed = 0

    def inject_bit_flip(self, byte_index: int, bit_index: int) -> None:
        """Simulates a radiation heavy ion strike flipping a bit in ECC memory."""
        if not (0 <= byte_index < len(self.ecc_buffer)):
            raise IndexError("Byte index out of bounds")
        if not (0 <= bit_index < 8):
            raise ValueError("Bit index must be 0..7")
        self.ecc_buffer[byte_index] ^= (1 << bit_index)

    def scrub_cycle(self) -> Dict[str, Any]:
        """Performs a full memory sweep, repairing in-place any single-bit SEUs."""
        corrections_this_cycle = 0
        uncorrectable_this_cycle = 0

        for i in range(len(self.ecc_buffer) // 2):
            lo_idx = 2 * i
            hi_idx = 2 * i + 1

            lo_c = self.ecc_buffer[lo_idx]
            hi_c = self.ecc_buffer[hi_idx]

            lo_nib, st_lo = HammingSECDED.decode_nibble(lo_c)
            hi_nib, st_hi = HammingSECDED.decode_nibble(hi_c)

            # In-place write-back of corrected codewords
            if st_lo == ECCStatus.SINGLE_ERROR_CORRECTED:
                self.ecc_buffer[lo_idx] = HammingSECDED.encode_nibble(lo_nib)
                corrections_this_cycle += 1
            elif st_lo == ECCStatus.DOUBLE_ERROR_DETECTED:
                uncorrectable_this_cycle += 1

            if st_hi == ECCStatus.SINGLE_ERROR_CORRECTED:
                self.ecc_buffer[hi_idx] = HammingSECDED.encode_nibble(hi_nib)
                corrections_this_cycle += 1
            elif st_hi == ECCStatus.DOUBLE_ERROR_DETECTED:
                uncorrectable_this_cycle += 1

        self.total_single_flips_corrected += corrections_this_cycle
        self.total_double_flips_detected += uncorrectable_this_cycle
        self.scrub_cycles_completed += 1

        return {
            "scrub_cycles": self.scrub_cycles_completed,
            "corrections_made": corrections_this_cycle,
            "uncorrectable_errors": uncorrectable_this_cycle,
            "total_corrected": self.total_single_flips_corrected,
        }

    def read_clean_data(self) -> Tuple[bytes, ECCStatus]:
        """Reads scrubbed memory and returns decoded payload."""
        data, status, _ = HammingSECDED.decode_and_correct_bytes(bytes(self.ecc_buffer))
        return data, status


@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Quaternion:
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            return cls(1.0, 0.0, 0.0, 0.0)
        unit = arr / norm
        return cls(float(unit[0]), float(unit[1]), float(unit[2]), float(unit[3]))

    def distance_to(self, other: Quaternion) -> float:
        """Geodesic distance between two unit quaternions: d(p, q) = 2 * arccos(|<p, q>|)."""
        p = self.to_array()
        q = other.to_array()
        dot = abs(float(np.dot(p, q)))
        dot = min(1.0, max(-1.0, dot))
        return 2.0 * math.acos(dot)


class TripleModularAttitudeVoting:
    """
    Triple-Modular Redundancy (TMR) Voting Engine for UAV attitude state vectors.
    Evaluates 3 redundant attitude quaternion registers (q0, q1, q2),
    detects single-event upsets via geodesic sphere distance, votes majority consensus,
    and automatically self-heals corrupted registers.
    """

    def __init__(self, initial_q: Optional[Quaternion] = None, max_discrepancy_rad: float = 0.05):
        if initial_q is None:
            initial_q = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.reg_a = initial_q
        self.reg_b = initial_q
        self.reg_c = initial_q
        self.max_discrepancy = max_discrepancy_rad
        self.seu_recoveries_count = 0
        self.critical_divergence_count = 0

    def update_state(self, q: Quaternion) -> None:
        """Synchronously updates all three redundant registers."""
        normalized = Quaternion.from_array(q.to_array())
        self.reg_a = normalized
        self.reg_b = normalized
        self.reg_c = normalized

    def inject_seu_flip(self, register_name: str, axis: str, delta: float) -> None:
        """Injects an SEU bit-flip / drift perturbation into a specific register."""
        target = getattr(self, f"reg_{register_name}")
        arr = target.to_array()
        axis_map = {"w": 0, "x": 1, "y": 2, "z": 3}
        arr[axis_map[axis]] += delta
        setattr(self, f"reg_{register_name}", Quaternion.from_array(arr))

    def vote_and_repair(self) -> Tuple[Quaternion, bool]:
        """
        Executes majority voting:
        - If all 3 agree: return average quaternion.
        - If 2 agree and 1 is corrupted: return consensus of the 2, and self-heal the corrupted register.
        - If all 3 diverge: return reg_a and flag critical fault.
        Returns: (consensus_quaternion, is_repaired_or_healthy)
        """
        d_ab = self.reg_a.distance_to(self.reg_b)
        d_bc = self.reg_b.distance_to(self.reg_c)
        d_ca = self.reg_c.distance_to(self.reg_a)

        tol = self.max_discrepancy

        # Case 1: All 3 agree
        if d_ab <= tol and d_bc <= tol and d_ca <= tol:
            avg_arr = (self.reg_a.to_array() + self.reg_b.to_array() + self.reg_c.to_array()) / 3.0
            consensus = Quaternion.from_array(avg_arr)
            return consensus, True

        # Case 2: A and B agree, C corrupted
        if d_ab <= tol and (d_bc > tol or d_ca > tol):
            avg_arr = (self.reg_a.to_array() + self.reg_b.to_array()) / 2.0
            consensus = Quaternion.from_array(avg_arr)
            # Self-healing: Repair Register C
            self.reg_c = consensus
            self.seu_recoveries_count += 1
            return consensus, True

        # Case 3: B and C agree, A corrupted
        if d_bc <= tol and (d_ab > tol or d_ca > tol):
            avg_arr = (self.reg_b.to_array() + self.reg_c.to_array()) / 2.0
            consensus = Quaternion.from_array(avg_arr)
            # Self-healing: Repair Register A
            self.reg_a = consensus
            self.seu_recoveries_count += 1
            return consensus, True

        # Case 4: C and A agree, B corrupted
        if d_ca <= tol and (d_ab > tol or d_bc > tol):
            avg_arr = (self.reg_c.to_array() + self.reg_a.to_array()) / 2.0
            consensus = Quaternion.from_array(avg_arr)
            # Self-healing: Repair Register B
            self.reg_b = consensus
            self.seu_recoveries_count += 1
            return consensus, True

        # Case 5: Critical Divergence (all 3 differ)
        self.critical_divergence_count += 1
        return self.reg_a, False


class CrowbarState(str, Enum):
    ARMED = "ARMED"
    TRIPPED = "TRIPPED"
    COOLING_DOWN = "COOLING_DOWN"
    RECOVERY = "RECOVERY"


class OvercurrentCrowbarController:
    """
    Ultra-Fast Electronic Crowbar Circuit & Overcurrent Latch Firmware.
    Monitors current sensor feedback; trips thyristor/MOSFET shunt when current
    exceeds critical limits to protect sensitive SoC core silicon from cosmic ray latch-ups.
    """

    def __init__(
        self,
        nominal_current_a: float = 1.5,
        trip_threshold_a: float = 3.5,
        hard_surge_threshold_a: float = 6.0,
        cooldown_period_s: float = 0.5,
    ):
        self.nominal_current = nominal_current_a
        self.trip_threshold = trip_threshold_a
        self.hard_surge = hard_surge_threshold_a
        self.cooldown_period_s = cooldown_period_s

        self.state = CrowbarState.ARMED
        self.tripped_timestamp: Optional[float] = None
        self.trip_history: List[Dict[str, Any]] = []

    def evaluate_current(self, current_amperes: float, current_time_s: Optional[float] = None) -> CrowbarState:
        """Evaluates current measurement and executes crowbar latching if overloaded."""
        if current_time_s is None:
            current_time_s = time.time()

        if self.state == CrowbarState.TRIPPED:
            # Check cooldown
            if (current_time_s - self.tripped_timestamp) >= self.cooldown_period_s:
                self.state = CrowbarState.COOLING_DOWN
            return self.state

        if self.state == CrowbarState.COOLING_DOWN:
            if current_amperes <= self.nominal_current:
                self.state = CrowbarState.ARMED
            return self.state

        # State is ARMED
        if current_amperes >= self.trip_threshold:
            self.state = CrowbarState.TRIPPED
            self.tripped_timestamp = current_time_s
            self.trip_history.append({
                "timestamp": current_time_s,
                "current_a": current_amperes,
                "type": "HARD_SURGE" if current_amperes >= self.hard_surge else "OVERCURRENT_TRIP",
            })

        return self.state

    def reset_latch(self) -> bool:
        """Manual software reset command."""
        self.state = CrowbarState.ARMED
        self.tripped_timestamp = None
        return True
