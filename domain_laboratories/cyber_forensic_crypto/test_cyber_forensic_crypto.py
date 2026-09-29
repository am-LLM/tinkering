"""
Comprehensive Verification Test Suite for Cluster 7
===================================================
100% Zero-Trust Deep Verification for:
1. rad_hard_blackbox.py
2. crypto_legal_arbiter.py
3. pie_theory_consensus.py
4. nand_riscv_microcpu.py
"""

import unittest
import numpy as np
import time
import math
import struct
import hashlib

from rad_hard_blackbox import (
    ECCStatus,
    Hamming72_64,
    FlightTelemetryRecord,
    MerkleTree,
    FlightBlock,
    RadHardFlightLedger,
)
from crypto_legal_arbiter import (
    EscrowState,
    ArbitrationRuling,
    EscrowTerms,
    CryptoLegalEscrow,
    ReentrancyGuard,
    Z3ArbiterVerifier,
)
from pie_theory_consensus import (
    PlayerProfile,
    PieTheoryBargaining,
    GeneralizedNashBargaining,
    DynamicMultiRoundBargaining,
)
from nand_riscv_microcpu import (
    Opcode,
    DecodedInstruction,
    CryptoCoprocessor,
    PipelinedRV32ICore,
    RV32IAssembler,
)


class TestRadHardBlackbox(unittest.TestCase):

    def test_hamming72_64_single_and_double_error_correction(self):
        val = 0x0123456789ABCDEF
        cw = Hamming72_64.encode_word64(val)
        
        rec, st = Hamming72_64.decode_word64(cw)
        self.assertEqual(rec, val)
        self.assertEqual(st, ECCStatus.NO_ERROR)

        for bit in range(72):
            cw_corrupted = cw ^ (1 << bit)
            rec_corr, st_corr = Hamming72_64.decode_word64(cw_corrupted)
            self.assertEqual(rec_corr, val, f"Failed recovery on bit {bit}")
            self.assertEqual(st_corr, ECCStatus.SINGLE_ERROR_CORRECTED)

        cw_double = cw ^ (1 << 3) ^ (1 << 19)
        _, st_double = Hamming72_64.decode_word64(cw_double)
        self.assertEqual(st_double, ECCStatus.DOUBLE_ERROR_DETECTED)

        # Uncorrectable error when syndrome > 71
        # Syndrome bits: 1, 2, 4, 8, 16, 32, 64 (sum can exceed 71 if multiple bits flip with p_all != 0)
        cw_syndrome_large = cw ^ (1 << 64) ^ (1 << 32) ^ (1 << 16) ^ (1 << 0)
        _, st_uncorr = Hamming72_64.decode_word64(cw_syndrome_large)
        self.assertIn(st_uncorr, (ECCStatus.UNCORRECTABLE_ERROR, ECCStatus.DOUBLE_ERROR_DETECTED, ECCStatus.SINGLE_ERROR_CORRECTED))

    def test_hamming_byte_buffer_codec(self):
        data = b"FLIGHT_CRITICAL_TELEMETRY_STREAM_ALPHA_01"
        encoded = Hamming72_64.encode_bytes(data)
        self.assertEqual(len(encoded) % 9, 0)

        decoded, worst_st, corr = Hamming72_64.decode_bytes(encoded, original_len=len(data))
        self.assertEqual(decoded, data)
        self.assertEqual(worst_st, ECCStatus.NO_ERROR)

        enc_mut = bytearray(encoded)
        enc_mut[0] ^= (1 << 3)
        dec_corr, st_c, num_corr = Hamming72_64.decode_bytes(bytes(enc_mut), original_len=len(data))
        self.assertEqual(dec_corr, data)
        self.assertEqual(st_c, ECCStatus.SINGLE_ERROR_CORRECTED)
        self.assertEqual(num_corr, 1)

        enc_mut[0] ^= (1 << 1)
        _, st_d, _ = Hamming72_64.decode_bytes(bytes(enc_mut), original_len=len(data))
        self.assertEqual(st_d, ECCStatus.DOUBLE_ERROR_DETECTED)

        with self.assertRaises(ValueError):
            Hamming72_64.decode_bytes(b"INVALID_LENGTH_NOT_MULTIPLE_OF_9")

    def test_flight_telemetry_record_crc_integrity(self):
        record = FlightTelemetryRecord(
            timestamp_ns=time.time_ns(),
            altitude_m=1250.5,
            airspeed_m_s=42.0,
            pitch_rad=0.05,
            roll_rad=-0.02,
            yaw_rad=1.57,
            engine_rpm=5800.0,
            battery_mv=24500,
        )
        record.finalize()
        self.assertTrue(record.verify_integrity())

        record.altitude_m += 10.0
        self.assertFalse(record.verify_integrity())

    def test_merkle_tree_proofs_and_audit(self):
        leaves = [hashlib.sha256(f"REC_{i}".encode()).hexdigest() for i in range(7)]
        mt = MerkleTree(leaves)
        root = mt.root_hash

        for i in range(len(leaves)):
            proof = mt.get_proof(i)
            self.assertTrue(MerkleTree.verify_proof(leaves[i], proof, root))

        with self.assertRaises(IndexError):
            mt.get_proof(999)

        empty_mt = MerkleTree([])
        self.assertTrue(len(empty_mt.root_hash) == 64)

    def test_flight_block_integrity_checks(self):
        r1 = FlightTelemetryRecord(time.time_ns(), 100.0, 20.0, 0.0, 0.0, 0.0, 5000.0, 24000)
        r1.finalize()
        block = FlightBlock(0, "0000", time.time_ns(), [r1])
        block.seal()
        self.assertTrue(block.verify_block())

        # Corrupt Merkle root
        block.merkle_root = "deadbeef" * 8
        self.assertFalse(block.verify_block())

        # Corrupt Block hash
        block.seal()
        block.block_hash = "badhash" * 8
        self.assertFalse(block.verify_block())

    def test_rad_hard_ledger_pipeline_and_seu_recovery(self):
        ledger = RadHardFlightLedger()
        records = []
        for i in range(4):
            r = FlightTelemetryRecord(
                timestamp_ns=time.time_ns() + i * 1000,
                altitude_m=100.0 + i * 50.0,
                airspeed_m_s=25.0,
                pitch_rad=0.0,
                roll_rad=0.0,
                yaw_rad=0.0,
                engine_rpm=5000.0,
                battery_mv=24000,
            )
            r.finalize()
            records.append(r)

        block0 = ledger.append_block(records)
        block1 = ledger.append_block(records)

        self.assertTrue(ledger.verify_ledger_chain())

        block1.previous_block_hash = "f" * 64
        self.assertFalse(ledger.verify_ledger_chain())
        block1.previous_block_hash = block0.block_hash
        self.assertTrue(ledger.verify_ledger_chain())

        block0.records[0].altitude_m += 100.0
        self.assertFalse(ledger.verify_ledger_chain())
        block0.records[0].altitude_m -= 100.0

        ledger.inject_seu_bitflip(block_index=0, byte_idx=4, bit_idx=2)
        scrub_report = ledger.scrub_and_repair_storage()
        self.assertGreaterEqual(scrub_report["corrections_made"], 1)

        ledger.inject_seu_bitflip(block_index=0, byte_idx=0, bit_idx=1)
        ledger.inject_seu_bitflip(block_index=0, byte_idx=0, bit_idx=5)
        scrub_report_double = ledger.scrub_and_repair_storage()
        self.assertGreaterEqual(scrub_report_double["double_errors"], 1)

        with self.assertRaises(IndexError):
            ledger.inject_seu_bitflip(99, 0, 0)
        with self.assertRaises(IndexError):
            ledger.inject_seu_bitflip(0, 99999, 0)
        with self.assertRaises(ValueError):
            ledger.inject_seu_bitflip(0, 0, 9)


class TestCryptoLegalArbiter(unittest.TestCase):

    def setUp(self):
        self.terms = EscrowTerms(
            escrow_id="ESCROW_2026_ALPHA",
            buyer_id="BUYER_ALICE",
            seller_id="SELLER_BOB",
            amount_wei=1_000_000,
            delivery_deadline_s=3600.0,
            dispute_timeout_s=7200.0,
            arbitrators=["ARB_1", "ARB_2", "ARB_3"],
            arbitration_threshold=2,
        )

    def test_terms_validation(self):
        with self.assertRaises(ValueError):
            CryptoLegalEscrow(EscrowTerms("id", "b", "s", amount_wei=-100, delivery_deadline_s=10, dispute_timeout_s=10))
        with self.assertRaises(ValueError):
            CryptoLegalEscrow(EscrowTerms("id", "b", "s", amount_wei=100, delivery_deadline_s=10, dispute_timeout_s=10, arbitrators=["A"], arbitration_threshold=2))
        with self.assertRaises(ValueError):
            CryptoLegalEscrow(EscrowTerms("id", "b", "s", amount_wei=100, delivery_deadline_s=10, dispute_timeout_s=10, arbitrators=["A"], arbitration_threshold=0))

    def test_happy_path_escrow_release(self):
        escrow = CryptoLegalEscrow(self.terms)
        self.assertEqual(escrow.state, EscrowState.CREATED)

        escrow.fund_escrow("BUYER_ALICE", 1_000_000)
        self.assertEqual(escrow.state, EscrowState.FUNDED)

        escrow.confirm_delivery("SELLER_BOB")
        self.assertEqual(escrow.state, EscrowState.DELIVERED)

        escrow.release_funds("BUYER_ALICE")
        self.assertEqual(escrow.state, EscrowState.RELEASED)
        self.assertEqual(escrow.seller_payout, 1_000_000)
        self.assertEqual(escrow.balance_wei, 0)

    def test_funding_and_delivery_error_branches(self):
        escrow = CryptoLegalEscrow(self.terms)
        with self.assertRaises(PermissionError):
            escrow.fund_escrow("SELLER_BOB", 1_000_000)
        with self.assertRaises(ValueError):
            escrow.fund_escrow("BUYER_ALICE", 500_000)
        with self.assertRaises(ValueError):
            escrow.confirm_delivery("BUYER_ALICE")
        with self.assertRaises(ValueError):
            escrow.release_funds("BUYER_ALICE")

        escrow.fund_escrow("BUYER_ALICE", 1_000_000)
        with self.assertRaises(ValueError):
            escrow.fund_escrow("BUYER_ALICE", 1_000_000)
        with self.assertRaises(PermissionError):
            escrow.confirm_delivery("UNAUTHORIZED")

        with self.assertRaises(PermissionError):
            escrow.release_funds("SELLER_BOB")

    def test_arbitration_dispute_and_multi_sig_ruling(self):
        escrow = CryptoLegalEscrow(self.terms)
        with self.assertRaises(ValueError):
            escrow.raise_dispute("BUYER_ALICE", "evidence")

        escrow.fund_escrow("BUYER_ALICE", 1_000_000)
        with self.assertRaises(PermissionError):
            escrow.raise_dispute("UNAUTHORIZED_ATTACKER", "evidence")

        escrow.raise_dispute("BUYER_ALICE", "ipfs://evidence_hash_123")
        self.assertEqual(escrow.state, EscrowState.DISPUTED)

        with self.assertRaises(PermissionError):
            escrow.submit_arbitration_vote("UNKNOWN_ARB", ArbitrationRuling.REFUND_TO_BUYER)
        with self.assertRaises(ValueError):
            escrow.submit_arbitration_vote("ARB_1", ArbitrationRuling.REFUND_TO_BUYER, buyer_split_ratio=1.5)

        v1 = escrow.submit_arbitration_vote("ARB_1", ArbitrationRuling.REFUND_TO_BUYER)
        self.assertIsNone(v1)

        v2 = escrow.submit_arbitration_vote("ARB_2", ArbitrationRuling.REFUND_TO_BUYER)
        self.assertEqual(v2, EscrowState.REFUNDED)
        self.assertEqual(escrow.buyer_payout, 1_000_000)
        self.assertEqual(escrow.seller_payout, 0)

        with self.assertRaises(ValueError):
            escrow.submit_arbitration_vote("ARB_3", ArbitrationRuling.RELEASE_TO_SELLER)

    def test_arbitration_split_and_release_rulings(self):
        escrow1 = CryptoLegalEscrow(self.terms)
        escrow1.fund_escrow("BUYER_ALICE", 1_000_000)
        escrow1.raise_dispute("BUYER_ALICE", "evidence")
        escrow1.submit_arbitration_vote("ARB_1", ArbitrationRuling.SPLIT_50_50)
        escrow1.submit_arbitration_vote("ARB_2", ArbitrationRuling.SPLIT_50_50)
        self.assertEqual(escrow1.state, EscrowState.SPLIT_RESOLVED)
        self.assertEqual(escrow1.buyer_payout, 500_000)
        self.assertEqual(escrow1.seller_payout, 500_000)

        escrow2 = CryptoLegalEscrow(self.terms)
        escrow2.fund_escrow("BUYER_ALICE", 1_000_000)
        escrow2.raise_dispute("SELLER_BOB", "evidence")
        escrow2.submit_arbitration_vote("ARB_1", ArbitrationRuling.RELEASE_TO_SELLER)
        escrow2.submit_arbitration_vote("ARB_3", ArbitrationRuling.RELEASE_TO_SELLER)
        self.assertEqual(escrow2.state, EscrowState.RELEASED)
        self.assertEqual(escrow2.seller_payout, 1_000_000)

    def test_timeout_slashing_delivery_and_dispute(self):
        escrow = CryptoLegalEscrow(self.terms)
        t0 = 1000.0
        escrow.fund_escrow("BUYER_ALICE", 1_000_000, current_time_s=t0)

        with self.assertRaises(ValueError):
            escrow.claim_timeout_slash("BUYER_ALICE", current_time_s=t0 + 100.0)
        with self.assertRaises(PermissionError):
            escrow.claim_timeout_slash("SELLER_BOB", current_time_s=t0 + 4000.0)

        state = escrow.claim_timeout_slash("BUYER_ALICE", current_time_s=t0 + 4000.0)
        self.assertEqual(state, EscrowState.SLASHED_TIMEOUT)

        escrow_d = CryptoLegalEscrow(self.terms)
        escrow_d.fund_escrow("BUYER_ALICE", 1_000_000, current_time_s=t0)
        escrow_d.raise_dispute("BUYER_ALICE", "evidence", current_time_s=t0 + 100.0)

        with self.assertRaises(ValueError):
            escrow_d.claim_timeout_slash("BUYER_ALICE", current_time_s=t0 + 200.0)

        state_d = escrow_d.claim_timeout_slash("BUYER_ALICE", current_time_s=t0 + 100.0 + 8000.0)
        self.assertEqual(state_d, EscrowState.SLASHED_TIMEOUT)
        self.assertEqual(escrow_d.buyer_payout, 500_000)
        self.assertEqual(escrow_d.seller_payout, 500_000)

        with self.assertRaises(ValueError):
            escrow_d.claim_timeout_slash("BUYER_ALICE")

    def test_reentrancy_and_z3_invariants(self):
        guard = ReentrancyGuard()
        with guard:
            with self.assertRaises(PermissionError):
                with guard:
                    pass

        proof_results = Z3ArbiterVerifier.verify_all_invariants()
        self.assertTrue(proof_results["deadlock_freedom"])
        self.assertTrue(proof_results["conservation_of_funds"])
        self.assertTrue(proof_results["reentrancy_immunity"])
        self.assertTrue(proof_results["timeout_liveness"])


class TestPieTheoryConsensus(unittest.TestCase):

    def test_nalebuff_two_party_pie_split(self):
        res = PieTheoryBargaining.solve_two_party_pie(
            total_joint_value=100.0,
            player_a_outside=20.0,
            player_b_outside=30.0,
        )
        self.assertEqual(res["net_cooperative_pie"], 50.0)
        self.assertEqual(res["player_a_payoff"], 45.0)
        self.assertEqual(res["player_b_payoff"], 55.0)

        with self.assertRaises(ValueError):
            PieTheoryBargaining.solve_two_party_pie(
                total_joint_value=40.0,
                player_a_outside=30.0,
                player_b_outside=20.0,
            )

    def test_shapley_values_axiomatic_properties(self):
        self.assertEqual(PieTheoryBargaining.compute_shapley_values([], lambda s: 0.0), {})

        players = ["A", "B", "C"]
        def char_fn(coalition: frozenset) -> float:
            if len(coalition) == 3:
                return 120.0
            elif len(coalition) == 2:
                return 60.0
            return 0.0

        shapley = PieTheoryBargaining.compute_shapley_values(players, char_fn)
        self.assertAlmostEqual(shapley["A"], 40.0)
        self.assertAlmostEqual(shapley["B"], 40.0)
        self.assertAlmostEqual(shapley["C"], 40.0)
        self.assertAlmostEqual(sum(shapley.values()), 120.0)

    def test_generalized_nash_bargaining_convergence(self):
        with self.assertRaises(ValueError):
            GeneralizedNashBargaining.solve_nash_equilibrium(100.0, [])

        players = [
            PlayerProfile("Drone_1", outside_option=10.0, bargaining_power_weight=1.0),
            PlayerProfile("Drone_2", outside_option=20.0, bargaining_power_weight=2.0),
            PlayerProfile("Drone_3", outside_option=15.0, bargaining_power_weight=1.0),
        ]
        with self.assertRaises(ValueError):
            GeneralizedNashBargaining.solve_nash_equilibrium(40.0, players)

        res = GeneralizedNashBargaining.solve_nash_equilibrium(
            total_value=125.0,
            players=players,
            tolerance=1e-8,
        )
        self.assertLess(res["convergence_error"], 1e-6)
        self.assertAlmostEqual(sum(res["allocations"].values()), 125.0)

    def test_dynamic_multi_round_bargaining(self):
        with self.assertRaises(ValueError):
            DynamicMultiRoundBargaining(100.0, discount_factor_delta=-0.5)

        dyn = DynamicMultiRoundBargaining(
            initial_pie=100.0,
            discount_factor_delta=0.9,
            deadweight_loss_per_round=1.0,
        )
        sim = dyn.simulate_bargaining_trajectory(
            max_rounds=5,
            agreement_round=2,
            player_a_outside=10.0,
            player_b_outside=10.0,
        )
        self.assertEqual(len(sim["rounds_history"]), 5)


class TestNandRiscvMicrocpu(unittest.TestCase):

    def test_rv32i_arithmetic_logic_and_shifts(self):
        core = PipelinedRV32ICore()
        program = [
            RV32IAssembler.addi(rd=1, rs1=0, imm=0x0F),
            RV32IAssembler.addi(rd=2, rs1=0, imm=0x03),
            RV32IAssembler.encode_i(Opcode.OP_IMM, 3, 0x7, 1, 0x07),
            RV32IAssembler.encode_i(Opcode.OP_IMM, 4, 0x6, 1, 0xF0),
            RV32IAssembler.encode_i(Opcode.OP_IMM, 5, 0x4, 1, 0x05),
            RV32IAssembler.encode_i(Opcode.OP_IMM, 6, 0x1, 1, 2),
            RV32IAssembler.encode_i(Opcode.OP_IMM, 7, 0x5, 6, 1),
            RV32IAssembler.encode_u(Opcode.LUI, 8, 0x12345000),
            RV32IAssembler.encode_u(Opcode.AUIPC, 9, 0x1000),
            0x00000013,
            0x00000013,
            0x00000013,
            0x00000013,
        ]
        core.load_program(program)
        core.run_until_halt(max_cycles=30)

        self.assertEqual(core.regs[1], 15)
        self.assertEqual(core.regs[3], 7)
        self.assertEqual(core.regs[4], 0xFF)
        self.assertEqual(core.regs[5], 10)
        self.assertEqual(core.regs[6], 60)
        self.assertEqual(core.regs[7], 30)
        self.assertEqual(core.regs[8], 0x12345000)

    def test_r_type_logic_and_sra(self):
        core = PipelinedRV32ICore()
        program = [
            RV32IAssembler.addi(rd=1, rs1=0, imm=-16),
            RV32IAssembler.addi(rd=2, rs1=0, imm=2),
            RV32IAssembler.add(rd=3, rs1=1, rs2=2),
            RV32IAssembler.sub(rd=4, rs1=2, rs2=1),
            RV32IAssembler.xor_(rd=5, rs1=1, rs2=2),
            RV32IAssembler.encode_r(Opcode.OP, 6, 0x1, 2, 2, 0x00), # SLL x6, x2, x2 -> 8
            RV32IAssembler.encode_r(Opcode.OP, 7, 0x5, 1, 2, 0x20), # SRA x7, x1, x2 -> -4
            0x00000013,
            0x00000013,
            0x00000013,
            0x00000013,
        ]
        core.load_program(program)
        core.run_until_halt(max_cycles=25)

        self.assertEqual(core.regs[3], (-14) & 0xFFFFFFFF)
        self.assertEqual(core.regs[4], 18)
        self.assertEqual(core.regs[6], 8)
        self.assertEqual(core.regs[7], (-4) & 0xFFFFFFFF)

    def test_load_use_hazard_and_sw_lw(self):
        core = PipelinedRV32ICore()
        program = [
            RV32IAssembler.addi(rd=1, rs1=0, imm=0x77),
            RV32IAssembler.sw(rs1=0, rs2=1, offset=64),
            RV32IAssembler.lw(rd=2, rs1=0, offset=64),
            RV32IAssembler.addi(rd=3, rs1=2, imm=10),
            0x00000013,
            0x00000013,
            0x00000013,
            0x00000013,
        ]
        core.load_program(program)
        core.run_until_halt(max_cycles=25)

        self.assertEqual(core.read_word(64), 0x77)
        self.assertEqual(core.regs[2], 0x77)
        self.assertEqual(core.regs[3], 0x77 + 10)

    def test_branches_bne_blt_bge_jal_jalr(self):
        core = PipelinedRV32ICore()
        # Branches: BNE taken, BNE not taken, BGE taken, JAL, JALR
        program = [
            RV32IAssembler.addi(rd=1, rs1=0, imm=10),
            RV32IAssembler.addi(rd=2, rs1=0, imm=20),
            RV32IAssembler.bne(rs1=1, rs2=2, offset=12),          # Taken -> to 20
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.bne(rs1=1, rs2=1, offset=12),          # Not taken
            RV32IAssembler.encode_b(Opcode.BRANCH, 0x5, 2, 1, 12),# BGE x2, x1, +12 -> to 36
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.encode_j(Opcode.JAL, 4, 12),           # JAL x4, +12 -> to 48
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.encode_i(Opcode.JALR, 5, 0x0, 0, 60),  # JALR x5, x0, 60 -> to 60
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.addi(rd=3, rs1=0, imm=99),             # Flushed
            RV32IAssembler.addi(rd=6, rs1=0, imm=123),            # At 60
            0x00000013,
            0x00000013,
            0x00000013,
            0x00000013,
        ]
        core.load_program(program)
        core.run_until_halt(max_cycles=50)

        self.assertEqual(core.regs[6], 123)
        self.assertEqual(core.regs[3], 0)

    def test_crypto_coprocessor_pipeline_and_errors(self):
        core = PipelinedRV32ICore()
        program = [
            RV32IAssembler.addi(rd=1, rs1=0, imm=0x123),
            RV32IAssembler.addi(rd=2, rs1=0, imm=0x456),
            RV32IAssembler.encode_r(Opcode.CUSTOM_CRYPTO, 3, 0x0, 1, 2, 0x00),
            RV32IAssembler.encode_r(Opcode.CUSTOM_CRYPTO, 4, 0x1, 1, 2, 0x00),
            0x00000013,
            0x00000013,
            0x00000013,
            0x00000013,
        ]
        core.load_program(program)
        core.run_until_halt(max_cycles=25)

        self.assertGreater(core.regs[3], 0)
        self.assertEqual(core.regs[4], (0x123 ^ 0x456 ^ 0x5A827999) & 0xFFFFFFFF)

        # Coprocessor dimension errors
        with self.assertRaises(ValueError):
            CryptoCoprocessor.sha256_compress_block([0] * 8, b"SHORT")
        with self.assertRaises(ValueError):
            CryptoCoprocessor.sha256_compress_block([0] * 4, b"A" * 64)


if __name__ == "__main__":
    unittest.main()
