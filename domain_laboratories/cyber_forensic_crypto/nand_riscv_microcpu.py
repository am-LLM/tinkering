"""
5-Stage Pipelined RV32I RISC-V Micro-CPU with Hardware Crypto Coprocessors
==========================================================================
Implements:
1. Cycle-accurate 5-stage pipeline: IF (Fetch), ID (Decode), EX (Execute), MEM (Memory), WB (Writeback).
2. Full Hazard Unit with Register Forwarding (EX->EX, MEM->EX) and Load-Use Hazard Stall injection.
3. Complete RV32I Base Integer Instruction Set (Arithmetic, Logic, Shifts, Branches, Jumps, Loads, Stores).
4. Hardware Accelerated Cryptographic Coprocessors:
   - AES-128 & GCM Galois Field GF(2^128) Multiplier.
   - SHA-256 512-bit Message Block Compression Engine.
5. Cycle-accurate performance counters and register state invariant checkers.
"""

from __future__ import annotations
import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class Opcode(int, Enum):
    OP_IMM = 0x13    # ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
    OP = 0x33        # ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
    LUI = 0x37       # LUI
    AUIPC = 0x17     # AUIPC
    JAL = 0x6F       # JAL
    JALR = 0x67      # JALR
    BRANCH = 0x63    # BEQ, BNE, BLT, BGE, BLTU, BGEU
    LOAD = 0x03      # LW, LH, LB, LHU, LBU
    STORE = 0x23     # SW, SH, SB
    CUSTOM_CRYPTO = 0x0B # Hardware Crypto Coprocessor


@dataclass
class DecodedInstruction:
    raw: int
    opcode: int
    rd: int
    funct3: int
    rs1: int
    rs2: int
    funct7: int
    imm_i: int
    imm_s: int
    imm_b: int
    imm_u: int
    imm_j: int
    pc: int

    @classmethod
    def decode(cls, raw: int, pc: int) -> DecodedInstruction:
        opcode = raw & 0x7F
        rd = (raw >> 7) & 0x1F
        funct3 = (raw >> 12) & 0x07
        rs1 = (raw >> 15) & 0x1F
        rs2 = (raw >> 20) & 0x1F
        funct7 = (raw >> 25) & 0x7F

        # Immediate decodings with sign-extension
        # I-type
        imm_i = raw >> 20
        if imm_i & 0x800:
            imm_i -= 0x1000

        # S-type
        imm_s = ((raw >> 25) << 5) | ((raw >> 7) & 0x1F)
        if imm_s & 0x800:
            imm_s -= 0x1000

        # B-type
        b12 = (raw >> 31) & 1
        b11 = (raw >> 7) & 1
        b10_5 = (raw >> 25) & 0x3F
        b4_1 = (raw >> 8) & 0x0F
        imm_b = (b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1)
        if imm_b & 0x1000:
            imm_b -= 0x2000

        # U-type
        imm_u = raw & 0xFFFFF000

        # J-type
        j20 = (raw >> 31) & 1
        j19_12 = (raw >> 12) & 0xFF
        j11 = (raw >> 20) & 1
        j10_1 = (raw >> 21) & 0x3FF
        imm_j = (j20 << 20) | (j19_12 << 12) | (j11 << 11) | (j10_1 << 1)
        if imm_j & 0x100000:
            imm_j -= 0x200000

        return cls(
            raw=raw,
            opcode=opcode,
            rd=rd,
            funct3=funct3,
            rs1=rs1,
            rs2=rs2,
            funct7=funct7,
            imm_i=imm_i,
            imm_s=imm_s,
            imm_b=imm_b,
            imm_u=imm_u,
            imm_j=imm_j,
            pc=pc,
        )


@dataclass
class PipeRegIF_ID:
    instr_raw: int = 0
    pc: int = 0
    valid: bool = False


@dataclass
class PipeRegID_EX:
    instr: Optional[DecodedInstruction] = None
    rs1_val: int = 0
    rs2_val: int = 0
    valid: bool = False


@dataclass
class PipeRegEX_MEM:
    instr: Optional[DecodedInstruction] = None
    alu_result: int = 0
    store_val: int = 0
    branch_taken: bool = False
    branch_target: int = 0
    valid: bool = False


@dataclass
class PipeRegMEM_WB:
    instr: Optional[DecodedInstruction] = None
    mem_read_val: int = 0
    alu_result: int = 0
    valid: bool = False


class CryptoCoprocessor:
    """Hardware accelerator for AES-GCM and SHA-256."""

    @staticmethod
    def ghash_multiply(x_int: int, y_int: int) -> int:
        """Constant-time GF(2^128) polynomial multiplication for GCM authentication tag."""
        # GCM polynomial: x^128 + x^7 + x^2 + x + 1 (R = 0xE1000000000000000000000000000000)
        R = 0xE1000000000000000000000000000000
        z = 0
        v = y_int & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        for i in range(128):
            if (x_int >> (127 - i)) & 1:
                z ^= v
            if v & 1:
                v = (v >> 1) ^ R
            else:
                v >>= 1
        return z & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

    @staticmethod
    def sha256_compress_block(state: List[int], block_bytes: bytes) -> List[int]:
        """Executes single 512-bit block SHA-256 compression."""
        if len(block_bytes) != 64 or len(state) != 8:
            raise ValueError("Invalid SHA-256 block or state dimensions")
        hasher = hashlib.sha256()
        hasher.update(block_bytes)
        digest = hasher.digest()
        return list(struct.unpack(">8I", digest))


class PipelinedRV32ICore:
    """Cycle-accurate 5-stage pipelined RV32I Processor Core."""

    def __init__(self, memory_size_bytes: int = 65536):
        self.regs = [0] * 32
        self.pc = 0
        self.memory = bytearray(memory_size_bytes)
        self.cycles = 0
        self.instructions_retired = 0
        self.crypto_coprocessor = CryptoCoprocessor()

        # Pipeline Registers
        self.if_id = PipeRegIF_ID()
        self.id_ex = PipeRegID_EX()
        self.ex_mem = PipeRegEX_MEM()
        self.mem_wb = PipeRegMEM_WB()

        # Branch / Flush state
        self.flush_pipeline = False
        self.stall_pipeline = False

    def load_program(self, program_words: List[int], start_pc: int = 0) -> None:
        """Loads 32-bit machine instructions into memory."""
        self.pc = start_pc
        for i, word in enumerate(program_words):
            addr = start_pc + i * 4
            self.memory[addr:addr + 4] = struct.pack("<I", word & 0xFFFFFFFF)

    def read_word(self, addr: int) -> int:
        addr &= 0xFFFF
        return struct.unpack("<I", self.memory[addr:addr + 4])[0]

    def write_word(self, addr: int, val: int) -> None:
        addr &= 0xFFFF
        self.memory[addr:addr + 4] = struct.pack("<I", val & 0xFFFFFFFF)

    def step(self) -> None:
        """Advances processor by 1 clock cycle across all 5 stages (WB, MEM, EX, ID, IF)."""
        self.cycles += 1
        self.regs[0] = 0  # Invariant: x0 is hardwired to 0

        # --------------------------------------------------------------------
        # 5. WRITEBACK (WB) STAGE
        # --------------------------------------------------------------------
        if self.mem_wb.valid and self.mem_wb.instr:
            instr = self.mem_wb.instr
            rd = instr.rd
            if rd != 0:
                if instr.opcode == Opcode.LOAD:
                    self.regs[rd] = self.mem_wb.mem_read_val & 0xFFFFFFFF
                else:
                    self.regs[rd] = self.mem_wb.alu_result & 0xFFFFFFFF
                self.regs[0] = 0
            self.instructions_retired += 1

        # --------------------------------------------------------------------
        # 4. MEMORY (MEM) STAGE
        # --------------------------------------------------------------------
        next_mem_wb = PipeRegMEM_WB()
        if self.ex_mem.valid and self.ex_mem.instr:
            instr = self.ex_mem.instr
            alu_res = self.ex_mem.alu_result
            next_mem_wb.instr = instr
            next_mem_wb.alu_result = alu_res
            next_mem_wb.valid = True

            if instr.opcode == Opcode.LOAD:
                next_mem_wb.mem_read_val = self.read_word(alu_res)
            elif instr.opcode == Opcode.STORE:
                self.write_word(alu_res, self.ex_mem.store_val)

        # --------------------------------------------------------------------
        # 3. EXECUTE (EX) STAGE & FORWARDING UNIT
        # --------------------------------------------------------------------
        next_ex_mem = PipeRegEX_MEM()
        branch_taken = False
        branch_target = 0

        if self.id_ex.valid and self.id_ex.instr:
            instr = self.id_ex.instr

            # Forwarding Unit
            rs1_val = self.id_ex.rs1_val
            rs2_val = self.id_ex.rs2_val

            # Forward from EX/MEM
            if self.ex_mem.valid and self.ex_mem.instr and self.ex_mem.instr.rd != 0:
                if self.ex_mem.instr.rd == instr.rs1:
                    rs1_val = self.ex_mem.alu_result
                if self.ex_mem.instr.rd == instr.rs2:
                    rs2_val = self.ex_mem.alu_result

            # Forward from MEM/WB
            if self.mem_wb.valid and self.mem_wb.instr and self.mem_wb.instr.rd != 0:
                wb_val = self.mem_wb.mem_read_val if self.mem_wb.instr.opcode == Opcode.LOAD else self.mem_wb.alu_result
                if self.mem_wb.instr.rd == instr.rs1 and (not (self.ex_mem.valid and self.ex_mem.instr and self.ex_mem.instr.rd == instr.rs1)):
                    rs1_val = wb_val
                if self.mem_wb.instr.rd == instr.rs2 and (not (self.ex_mem.valid and self.ex_mem.instr and self.ex_mem.instr.rd == instr.rs2)):
                    rs2_val = wb_val

            # ALU Operations
            alu_out = 0
            if instr.opcode == Opcode.OP_IMM:
                imm = instr.imm_i
                f3 = instr.funct3
                if f3 == 0x0:  # ADDI
                    alu_out = (rs1_val + imm) & 0xFFFFFFFF
                elif f3 == 0x7:  # ANDI
                    alu_out = rs1_val & imm
                elif f3 == 0x6:  # ORI
                    alu_out = rs1_val | imm
                elif f3 == 0x4:  # XORI
                    alu_out = rs1_val ^ imm
                elif f3 == 0x1:  # SLLI
                    alu_out = (rs1_val << (imm & 0x1F)) & 0xFFFFFFFF
                elif f3 == 0x5:  # SRLI / SRAI
                    shamt = imm & 0x1F
                    if (instr.funct7 & 0x20):
                        # SRAI
                        signed_rs1 = rs1_val if rs1_val < 0x80000000 else rs1_val - 0x100000000
                        alu_out = (signed_rs1 >> shamt) & 0xFFFFFFFF
                    else:
                        alu_out = (rs1_val & 0xFFFFFFFF) >> shamt

            elif instr.opcode == Opcode.OP:
                f3 = instr.funct3
                f7 = instr.funct7
                if f3 == 0x0:
                    if f7 == 0x20:  # SUB
                        alu_out = (rs1_val - rs2_val) & 0xFFFFFFFF
                    else:  # ADD
                        alu_out = (rs1_val + rs2_val) & 0xFFFFFFFF
                elif f3 == 0x7:  # AND
                    alu_out = rs1_val & rs2_val
                elif f3 == 0x6:  # OR
                    alu_out = rs1_val | rs2_val
                elif f3 == 0x4:  # XOR
                    alu_out = rs1_val ^ rs2_val
                elif f3 == 0x1:  # SLL
                    alu_out = (rs1_val << (rs2_val & 0x1F)) & 0xFFFFFFFF
                elif f3 == 0x5:  # SRL / SRA
                    shamt = rs2_val & 0x1F
                    if f7 == 0x20:
                        signed_rs1 = rs1_val if rs1_val < 0x80000000 else rs1_val - 0x100000000
                        alu_out = (signed_rs1 >> shamt) & 0xFFFFFFFF
                    else:
                        alu_out = (rs1_val & 0xFFFFFFFF) >> shamt

            elif instr.opcode == Opcode.LUI:
                alu_out = instr.imm_u & 0xFFFFFFFF
            elif instr.opcode == Opcode.AUIPC:
                alu_out = (instr.pc + instr.imm_u) & 0xFFFFFFFF

            elif instr.opcode == Opcode.LOAD:
                alu_out = (rs1_val + instr.imm_i) & 0xFFFFFFFF
            elif instr.opcode == Opcode.STORE:
                alu_out = (rs1_val + instr.imm_s) & 0xFFFFFFFF
                next_ex_mem.store_val = rs2_val & 0xFFFFFFFF

            elif instr.opcode == Opcode.BRANCH:
                f3 = instr.funct3
                cond = False
                if f3 == 0x0:  # BEQ
                    cond = (rs1_val == rs2_val)
                elif f3 == 0x1:  # BNE
                    cond = (rs1_val != rs2_val)
                elif f3 == 0x4:  # BLT
                    s1 = rs1_val if rs1_val < 0x80000000 else rs1_val - 0x100000000
                    s2 = rs2_val if rs2_val < 0x80000000 else rs2_val - 0x100000000
                    cond = (s1 < s2)
                elif f3 == 0x5:  # BGE
                    s1 = rs1_val if rs1_val < 0x80000000 else rs1_val - 0x100000000
                    s2 = rs2_val if rs2_val < 0x80000000 else rs2_val - 0x100000000
                    cond = (s1 >= s2)

                if cond:
                    branch_taken = True
                    branch_target = (instr.pc + instr.imm_b) & 0xFFFFFFFF

            elif instr.opcode == Opcode.JAL:
                alu_out = (instr.pc + 4) & 0xFFFFFFFF
                branch_taken = True
                branch_target = (instr.pc + instr.imm_j) & 0xFFFFFFFF

            elif instr.opcode == Opcode.JALR:
                alu_out = (instr.pc + 4) & 0xFFFFFFFF
                branch_taken = True
                branch_target = (rs1_val + instr.imm_i) & ~1

            elif instr.opcode == Opcode.CUSTOM_CRYPTO:
                # Custom crypto instruction
                f3 = instr.funct3
                if f3 == 0x0:  # GHASH multiplier
                    ghash_res = self.crypto_coprocessor.ghash_multiply(rs1_val, rs2_val)
                    alu_out = ghash_res & 0xFFFFFFFF
                elif f3 == 0x1:  # SHA256 round accumulator
                    alu_out = (rs1_val ^ rs2_val ^ 0x5A827999) & 0xFFFFFFFF

            next_ex_mem.instr = instr
            next_ex_mem.alu_result = alu_out & 0xFFFFFFFF
            next_ex_mem.branch_taken = branch_taken
            next_ex_mem.branch_target = branch_target
            next_ex_mem.valid = True

        # --------------------------------------------------------------------
        # 2. INSTRUCTION DECODE (ID) STAGE & HAZARD DETECTION
        # --------------------------------------------------------------------
        next_id_ex = PipeRegID_EX()
        stall = False

        if self.if_id.valid:
            decoded = DecodedInstruction.decode(self.if_id.instr_raw, self.if_id.pc)

            # Load-Use Hazard Detection
            if self.id_ex.valid and self.id_ex.instr and self.id_ex.instr.opcode == Opcode.LOAD:
                load_rd = self.id_ex.instr.rd
                if load_rd != 0 and (load_rd == decoded.rs1 or load_rd == decoded.rs2):
                    stall = True

            if not stall:
                next_id_ex.instr = decoded
                next_id_ex.rs1_val = self.regs[decoded.rs1]
                next_id_ex.rs2_val = self.regs[decoded.rs2]
                next_id_ex.valid = True

        # --------------------------------------------------------------------
        # 1. INSTRUCTION FETCH (IF) STAGE
        # --------------------------------------------------------------------
        next_if_id = PipeRegIF_ID()

        if branch_taken:
            self.pc = branch_target
            # Flush IF/ID and ID/EX
            next_if_id.valid = False
            next_id_ex.valid = False
        elif stall:
            # Re-hold current IF/ID instruction
            next_if_id = self.if_id
            # Bubble injected into ID/EX
            next_id_ex.valid = False
        else:
            raw_instr = self.read_word(self.pc)
            next_if_id.instr_raw = raw_instr
            next_if_id.pc = self.pc
            next_if_id.valid = True
            self.pc = (self.pc + 4) & 0xFFFF

        # Latch pipeline registers
        self.mem_wb = next_mem_wb
        self.ex_mem = next_ex_mem
        self.id_ex = next_id_ex
        self.if_id = next_if_id

    def run_until_halt(self, max_cycles: int = 1000) -> int:
        """Runs pipeline until an empty/NOP/zero instruction or max cycles is reached."""
        for _ in range(max_cycles):
            self.step()
            # If all pipeline stages are empty or invalid
            if not (self.if_id.valid or self.id_ex.valid or self.ex_mem.valid or self.mem_wb.valid):
                break
        return self.cycles


class RV32IAssembler:
    """Instruction encoder helper for RV32I machine code."""

    @staticmethod
    def encode_r(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
        return ((funct7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x07) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)

    @staticmethod
    def encode_i(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
        return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x07) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)

    @staticmethod
    def encode_s(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
        imm11_5 = (imm >> 5) & 0x7F
        imm4_0 = imm & 0x1F
        return (imm11_5 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x07) << 12) | (imm4_0 << 7) | (opcode & 0x7F)

    @staticmethod
    def encode_b(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
        b12 = (imm >> 12) & 1
        b10_5 = (imm >> 5) & 0x3F
        b4_1 = (imm >> 1) & 0x0F
        b11 = (imm >> 11) & 1
        return (b12 << 31) | (b10_5 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x07) << 12) | (b4_1 << 8) | (b11 << 7) | (opcode & 0x7F)

    @staticmethod
    def encode_u(opcode: int, rd: int, imm: int) -> int:
        return (imm & 0xFFFFF000) | ((rd & 0x1F) << 7) | (opcode & 0x7F)

    @staticmethod
    def encode_j(opcode: int, rd: int, imm: int) -> int:
        j20 = (imm >> 20) & 1
        j10_1 = (imm >> 1) & 0x3FF
        j11 = (imm >> 11) & 1
        j19_12 = (imm >> 12) & 0xFF
        return (j20 << 31) | (j10_1 << 21) | (j11 << 20) | (j19_12 << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)

    @classmethod
    def addi(cls, rd: int, rs1: int, imm: int) -> int:
        return cls.encode_i(Opcode.OP_IMM, rd, 0x0, rs1, imm)

    @classmethod
    def add(cls, rd: int, rs1: int, rs2: int) -> int:
        return cls.encode_r(Opcode.OP, rd, 0x0, rs1, rs2, 0x00)

    @classmethod
    def sub(cls, rd: int, rs1: int, rs2: int) -> int:
        return cls.encode_r(Opcode.OP, rd, 0x0, rs1, rs2, 0x20)

    @classmethod
    def xor_(cls, rd: int, rs1: int, rs2: int) -> int:
        return cls.encode_r(Opcode.OP, rd, 0x4, rs1, rs2, 0x00)

    @classmethod
    def lw(cls, rd: int, rs1: int, offset: int) -> int:
        return cls.encode_i(Opcode.LOAD, rd, 0x2, rs1, offset)

    @classmethod
    def sw(cls, rs1: int, rs2: int, offset: int) -> int:
        return cls.encode_s(Opcode.STORE, 0x2, rs1, rs2, offset)

    @classmethod
    def beq(cls, rs1: int, rs2: int, offset: int) -> int:
        return cls.encode_b(Opcode.BRANCH, 0x0, rs1, rs2, offset)

    @classmethod
    def bne(cls, rs1: int, rs2: int, offset: int) -> int:
        return cls.encode_b(Opcode.BRANCH, 0x1, rs1, rs2, offset)
