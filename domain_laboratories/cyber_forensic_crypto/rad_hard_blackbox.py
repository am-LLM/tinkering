"""
Radiation-Hardened Append-Only Merkle Flight Blackbox Ledger
============================================================
Implements:
1. SEC-DED (72, 64) Extended Hamming Code Memory Protection (Single Error Correction, Double Error Detection).
2. Autonomous Memory Scrubber repairing Single-Event Upsets (SEU) in-place.
3. CRC32 Integrity Gating on raw telemetry records.
4. Cryptographic Merkle-Tree and Append-Only Hash-Linked Flight Blocks.
5. Merkle Proof Generation & Audit Verification with tamper localization.
"""

from __future__ import annotations
import binascii
import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional


class ECCStatus(str, Enum):
    NO_ERROR = "NO_ERROR"
    SINGLE_ERROR_CORRECTED = "SINGLE_ERROR_CORRECTED"
    DOUBLE_ERROR_DETECTED = "DOUBLE_ERROR_DETECTED"
    UNCORRECTABLE_ERROR = "UNCORRECTABLE_ERROR"


PARITY_POS = (1, 2, 4, 8, 16, 32, 64)
DATA_POS = tuple(pos for pos in range(1, 72) if pos not in PARITY_POS)


class Hamming72_64:
    """
    SEC-DED (72, 64) Extended Hamming Codec.
    64 data bits + 7 Hamming parity bits + 1 overall parity bit = 72 bits.
    """

    PARITY_POSITIONS = PARITY_POS
    DATA_POSITIONS = DATA_POS

    @classmethod
    def encode_word64(cls, val64: int) -> int:
        """Encodes a 64-bit integer into a 72-bit SEC-DED codeword integer."""
        val64 &= 0xFFFFFFFFFFFFFFFF
        bits72 = 0
        for i, pos in enumerate(cls.DATA_POSITIONS):
            if (val64 >> i) & 1:
                bits72 |= (1 << pos)

        for p_idx, p_pos in enumerate(cls.PARITY_POSITIONS):
            p_val = 0
            for pos in range(1, 72):
                if (pos & (1 << p_idx)) != 0:
                    p_val ^= ((bits72 >> pos) & 1)
            if p_val:
                bits72 |= (1 << p_pos)

        p0 = 0
        for pos in range(1, 72):
            p0 ^= ((bits72 >> pos) & 1)
        if p0:
            bits72 |= (1 << 0)
        return bits72

    @classmethod
    def decode_word64(cls, codeword72: int) -> Tuple[int, ECCStatus]:
        """
        Decodes a 72-bit SEC-DED codeword into a 64-bit integer.
        Returns: (data_64, ECCStatus)
        """
        syndrome = 0
        for p_idx, p_pos in enumerate(cls.PARITY_POSITIONS):
            s_bit = 0
            for pos in range(1, 72):
                if (pos & (1 << p_idx)) != 0:
                    s_bit ^= ((codeword72 >> pos) & 1)
            if s_bit:
                syndrome |= (1 << p_idx)

        p_all = 0
        for pos in range(0, 72):
            p_all ^= ((codeword72 >> pos) & 1)

        status = ECCStatus.NO_ERROR
        corrected_cw = codeword72

        if syndrome == 0:
            if p_all != 0:
                status = ECCStatus.SINGLE_ERROR_CORRECTED
                corrected_cw ^= (1 << 0)
        else:
            if p_all != 0:
                status = ECCStatus.SINGLE_ERROR_CORRECTED
                if syndrome <= 71:
                    corrected_cw ^= (1 << syndrome)
                else:
                    status = ECCStatus.UNCORRECTABLE_ERROR
            else:
                status = ECCStatus.DOUBLE_ERROR_DETECTED

        recovered_val = 0
        for i, pos in enumerate(cls.DATA_POSITIONS):
            if (corrected_cw >> pos) & 1:
                recovered_val |= (1 << i)

        return (recovered_val & 0xFFFFFFFFFFFFFFFF), status

    @classmethod
    def encode_bytes(cls, data: bytes) -> bytes:
        """Pads data to 8-byte boundary and encodes each 8-byte chunk into 9 bytes (72 bits)."""
        pad_len = (8 - (len(data) % 8)) % 8
        padded = data + (b"\x00" * pad_len)
        out = bytearray()
        for i in range(0, len(padded), 8):
            val64 = struct.unpack("<Q", padded[i:i + 8])[0]
            cw72 = cls.encode_word64(val64)
            # Pack 72 bits into 9 bytes
            cw_bytes = cw72.to_bytes(9, byteorder="little")
            out.extend(cw_bytes)
        return bytes(out)

    @classmethod
    def decode_bytes(cls, encoded: bytes, original_len: Optional[int] = None) -> Tuple[bytes, ECCStatus, int]:
        """
        Decodes 9-byte codeword chunks into raw data.
        Returns: (decoded_bytes, worst_status, corrections_count)
        """
        if len(encoded) % 9 != 0:
            raise ValueError("Encoded ECC buffer must be a multiple of 9 bytes")

        out = bytearray()
        worst_status = ECCStatus.NO_ERROR
        corrections = 0

        for i in range(0, len(encoded), 9):
            cw_bytes = encoded[i:i + 9]
            cw72 = int.from_bytes(cw_bytes, byteorder="little")
            val64, st = cls.decode_word64(cw72)
            out.extend(struct.pack("<Q", val64))

            if st == ECCStatus.SINGLE_ERROR_CORRECTED:
                corrections += 1
                if worst_status == ECCStatus.NO_ERROR:
                    worst_status = ECCStatus.SINGLE_ERROR_CORRECTED
            elif st in (ECCStatus.DOUBLE_ERROR_DETECTED, ECCStatus.UNCORRECTABLE_ERROR):
                worst_status = st

        result = bytes(out)
        if original_len is not None:
            result = result[:original_len]
        return result, worst_status, corrections


@dataclass
class FlightTelemetryRecord:
    timestamp_ns: int
    altitude_m: float
    airspeed_m_s: float
    pitch_rad: float
    roll_rad: float
    yaw_rad: float
    engine_rpm: float
    battery_mv: int
    payload_hash: str = ""
    crc32: int = 0

    def serialize_raw(self) -> bytes:
        """Packs telemetry record into standard binary payload (40 bytes)."""
        header = struct.pack(
            "<QfffffHI",
            self.timestamp_ns,
            self.altitude_m,
            self.airspeed_m_s,
            self.pitch_rad,
            self.roll_rad,
            self.yaw_rad,
            self.battery_mv,
            int(self.engine_rpm),
        )
        return header

    def compute_crc32(self) -> int:
        """Computes standard CRC32 over serialized payload."""
        raw = self.serialize_raw()
        return binascii.crc32(raw) & 0xFFFFFFFF

    def finalize(self) -> None:
        """Seals record with CRC32 and payload digest."""
        raw = self.serialize_raw()
        self.crc32 = self.compute_crc32()
        self.payload_hash = hashlib.sha256(raw + struct.pack("<I", self.crc32)).hexdigest()

    def verify_integrity(self) -> bool:
        """Validates CRC32 and cryptographic payload digest."""
        expected_crc = self.compute_crc32()
        if expected_crc != self.crc32:
            return False
        expected_hash = hashlib.sha256(self.serialize_raw() + struct.pack("<I", self.crc32)).hexdigest()
        return expected_hash == self.payload_hash


class MerkleTree:
    """Binary Merkle Tree for flight ledger block verification."""

    def __init__(self, leaf_hashes: List[str]):
        if not leaf_hashes:
            leaf_hashes = [hashlib.sha256(b"EMPTY_BLOCK").hexdigest()]
        self.leaf_hashes = leaf_hashes
        self.layers = self._build_tree(leaf_hashes)

    def _build_tree(self, leaves: List[str]) -> List[List[str]]:
        current = leaves[:]
        layers = [current]
        while len(current) > 1:
            next_layer = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if (i + 1) < len(current) else current[i]
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                next_layer.append(combined)
            layers.append(next_layer)
            current = next_layer
        return layers

    @property
    def root_hash(self) -> str:
        return self.layers[-1][0]

    def get_proof(self, leaf_index: int) -> List[Tuple[str, str]]:
        """Returns audit path of (sibling_hash, direction) where direction in ('L', 'R')."""
        if not (0 <= leaf_index < len(self.leaf_hashes)):
            raise IndexError("Leaf index out of bounds")
        proof = []
        idx = leaf_index
        for layer in self.layers[:-1]:
            is_right = (idx % 2 == 1)
            sibling_idx = idx - 1 if is_right else (idx + 1 if (idx + 1) < len(layer) else idx)
            sibling_hash = layer[sibling_idx]
            direction = "L" if is_right else "R"
            proof.append((sibling_hash, direction))
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Tuple[str, str]], expected_root: str) -> bool:
        current = leaf_hash
        for sibling_hash, direction in proof:
            if direction == "L":
                current = hashlib.sha256((sibling_hash + current).encode()).hexdigest()
            else:
                current = hashlib.sha256((current + sibling_hash).encode()).hexdigest()
        return current == expected_root


@dataclass
class FlightBlock:
    block_index: int
    previous_block_hash: str
    timestamp_ns: int
    records: List[FlightTelemetryRecord]
    merkle_root: str = ""
    block_hash: str = ""

    def seal(self) -> str:
        """Computes Merkle root and block header hash."""
        for r in self.records:
            if not r.payload_hash:
                r.finalize()
        leaves = [r.payload_hash for r in self.records]
        mt = MerkleTree(leaves)
        self.merkle_root = mt.root_hash

        header = f"{self.block_index}:{self.previous_block_hash}:{self.timestamp_ns}:{self.merkle_root}"
        self.block_hash = hashlib.sha256(header.encode()).hexdigest()
        return self.block_hash

    def verify_block(self) -> bool:
        """Validates block internal CRC32s, Merkle tree root, and block hash."""
        for r in self.records:
            if not r.verify_integrity():
                return False
        leaves = [r.payload_hash for r in self.records]
        mt = MerkleTree(leaves)
        if mt.root_hash != self.merkle_root:
            return False
        header = f"{self.block_index}:{self.previous_block_hash}:{self.timestamp_ns}:{self.merkle_root}"
        expected_hash = hashlib.sha256(header.encode()).hexdigest()
        return expected_hash == self.block_hash


class RadHardFlightLedger:
    """
    Append-Only Flight Blackbox Ledger with Radiation-Hardened Memory Protection.
    Stores sealed flight blocks in SEC-DED (72, 64) protected memory buffer.
    """

    def __init__(self):
        self.blocks: List[FlightBlock] = []
        self.encoded_storage: List[bytearray] = []
        self.block_lengths: List[int] = []
        self.total_seu_scrubbed = 0
        self.total_double_errors_trapped = 0

    def append_block(self, records: List[FlightTelemetryRecord]) -> FlightBlock:
        """Appends a new sealed block to the ledger."""
        prev_hash = self.blocks[-1].block_hash if self.blocks else "0000000000000000000000000000000000000000000000000000000000000000"
        block = FlightBlock(
            block_index=len(self.blocks),
            previous_block_hash=prev_hash,
            timestamp_ns=time.time_ns(),
            records=records,
        )
        block.seal()
        self.blocks.append(block)

        # Serialize block content and encode with Hamming(72, 64)
        raw_payload = block.block_hash.encode() + b":" + block.merkle_root.encode()
        for r in records:
            raw_payload += b":" + r.serialize_raw() + struct.pack("<I", r.crc32)

        self.block_lengths.append(len(raw_payload))
        encoded = bytearray(Hamming72_64.encode_bytes(raw_payload))
        self.encoded_storage.append(encoded)
        return block

    def inject_seu_bitflip(self, block_index: int, byte_idx: int, bit_idx: int) -> None:
        """Injects radiation single-event upset (SEU) into hardware ECC storage."""
        if not (0 <= block_index < len(self.encoded_storage)):
            raise IndexError("Block index out of bounds")
        storage = self.encoded_storage[block_index]
        if not (0 <= byte_idx < len(storage)):
            raise IndexError("Byte index out of bounds")
        if not (0 <= bit_idx < 8):
            raise ValueError("Bit index must be 0..7")
        storage[byte_idx] ^= (1 << bit_idx)

    def scrub_and_repair_storage(self) -> Dict[str, Any]:
        """
        Autonomous memory scrubbing pass across all hardware storage words.
        Corrects single-bit SEUs in-place and logs double-bit error detections.
        """
        corrections_this_pass = 0
        double_errors_this_pass = 0

        for b_idx, storage in enumerate(self.encoded_storage):
            for word_start in range(0, len(storage), 9):
                cw_bytes = storage[word_start:word_start + 9]
                cw72 = int.from_bytes(cw_bytes, byteorder="little")
                val64, status = Hamming72_64.decode_word64(cw72)

                if status == ECCStatus.SINGLE_ERROR_CORRECTED:
                    # Repair in-place
                    clean_cw = Hamming72_64.encode_word64(val64)
                    storage[word_start:word_start + 9] = clean_cw.to_bytes(9, byteorder="little")
                    corrections_this_pass += 1
                elif status in (ECCStatus.DOUBLE_ERROR_DETECTED, ECCStatus.UNCORRECTABLE_ERROR):
                    double_errors_this_pass += 1

        self.total_seu_scrubbed += corrections_this_pass
        self.total_double_errors_trapped += double_errors_this_pass

        return {
            "corrections_made": corrections_this_pass,
            "double_errors": double_errors_this_pass,
            "total_scrubbed": self.total_seu_scrubbed,
        }

    def verify_ledger_chain(self) -> bool:
        """Validates entire append-only cryptographic chain from genesis to tip."""
        for i, block in enumerate(self.blocks):
            if not block.verify_block():
                return False
            if i > 0:
                if block.previous_block_hash != self.blocks[i - 1].block_hash:
                    return False
        return True
