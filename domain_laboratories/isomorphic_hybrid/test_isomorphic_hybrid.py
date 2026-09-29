"""
Comprehensive Unit & Integration Test Suite
===========================================
100% Coverage Test Suite for:
1. isomorphic_physics_bridge.py
2. pqc_zk_swarm_consensus.py
3. silicon_side_channel_guard.py
4. seu_rad_hard_firmware.py
5. geo_acoustic_subterranean_slam.py
6. bio_myco_drone_airframe.py
"""

import unittest
import numpy as np
import time
import math
import hashlib

from isomorphic_physics_bridge import (
    IsomorphicPhysicsBridge,
    PhysicalDomain,
    InvariantSpectralSignature,
    CanonicalStateSpace,
)
from pqc_zk_swarm_consensus import (
    RingLWEPQC,
    PQCPostQuantumSigner,
    ZKTelemetryProver,
    ZKTelemetryVerifier,
    ZKTelemetryProof,
    SwarmNode,
    ConsensusState,
    ConsensusBlock,
    ZK_PRIME_P,
    ZK_ORDER,
    poly_add,
    poly_sub,
    sample_centered_binomial,
)
from silicon_side_channel_guard import (
    SecurityState,
    ConstantTimeOps,
    ConstantTimeAES,
    ConstantTimeRingArithmetic,
    ClockJitterEngine,
    VoltageGlitchDetector,
    HardwareSecurityGuard,
)
from seu_rad_hard_firmware import (
    ECCStatus,
    HammingSECDED,
    MemoryScrubber,
    Quaternion,
    TripleModularAttitudeVoting,
    CrowbarState,
    OvercurrentCrowbarController,
)
from geo_acoustic_subterranean_slam import (
    CrustalMagneticMap,
    UltrasonicPhasedArray,
    SubterraneanEKFSLAM,
)
from bio_myco_drone_airframe import (
    MycoChitinMaterial,
    MycoDroneFEMAirframe,
    BiodegradationSimulator,
)


class TestIsomorphicPhysicsBridge(unittest.TestCase):

    def setUp(self):
        self.bridge = IsomorphicPhysicsBridge(
            inertia=2.0,
            dissipation=0.4,
            elasticity=50.0,
            domain=PhysicalDomain.MECHANICAL
        )

    def test_spectral_invariants(self):
        sig = self.bridge.spectral_signature
        expected_omega_n = math.sqrt(50.0 / 2.0)
        expected_zeta = 0.4 / (2.0 * math.sqrt(50.0 * 2.0))
        expected_q = 1.0 / (2.0 * expected_zeta)

        self.assertAlmostEqual(sig.natural_frequency_rad_s, expected_omega_n, places=6)
        self.assertAlmostEqual(sig.damping_ratio, expected_zeta, places=6)
        self.assertAlmostEqual(sig.quality_factor, expected_q, places=6)
        self.assertTrue(sig.is_underdamped)
        self.assertTrue(sig.is_stable)
        self.assertIn("natural_frequency_rad_s", sig.to_dict())

    def test_overdamped_and_zero_damping(self):
        over_bridge = IsomorphicPhysicsBridge(inertia=1.0, dissipation=10.0, elasticity=4.0)
        sig_over = over_bridge.spectral_signature
        self.assertFalse(sig_over.is_underdamped)
        self.assertTrue(sig_over.is_stable)

        undamped = IsomorphicPhysicsBridge(inertia=1.0, dissipation=0.0, elasticity=100.0)
        sig_un = undamped.spectral_signature
        self.assertEqual(sig_un.quality_factor, float("inf"))
        self.assertEqual(sig_un.time_constant_s, float("inf"))

    def test_domain_mapping_all_domains(self):
        for target in [PhysicalDomain.MECHANICAL, PhysicalDomain.ELECTRICAL, PhysicalDomain.HYDRAULIC, PhysicalDomain.MACROECONOMIC]:
            mapped = self.bridge.map_to_target_domain(target, scaling_factor=2.0)
            self.assertEqual(mapped["target_domain"], target.value)
            self.assertIn("vocabulary", mapped)

        with self.assertRaises(ValueError):
            self.bridge.map_to_target_domain(PhysicalDomain.ELECTRICAL, scaling_factor=-1.0)
        with self.assertRaises(ValueError):
            self.bridge.map_to_target_domain("INVALID_DOMAIN")

    def test_state_space_and_transfer_function(self):
        ss = self.bridge.to_state_space()
        self.assertEqual(ss.A.shape, (2, 2))
        self.assertEqual(ss.B.shape, (2, 1))
        num, den = ss.compute_transfer_function()
        self.assertEqual(len(den), 3)

    def test_transient_simulation_and_forcing(self):
        res = self.bridge.simulate_transient_response(
            t_span=(0.0, 1.0),
            initial_state=(1.0, 0.0),
            forcing_function=lambda t: math.sin(2.0 * t),
            num_points=100
        )
        self.assertTrue(res["success"])
        self.assertEqual(len(res["time"]), 100)

    def test_diffusion_solver_and_cfl(self):
        u = self.bridge.solve_diffusion_isomorphism(diffusion_coeff=0.05, length=1.0, nx=50, nt=100)
        self.assertEqual(u.shape[1], 50)

        with self.assertRaises(ValueError):
            self.bridge.solve_diffusion_isomorphism(diffusion_coeff=-0.1, length=1.0)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            IsomorphicPhysicsBridge(inertia=0.0, dissipation=0.1, elasticity=10.0)
        with self.assertRaises(ValueError):
            IsomorphicPhysicsBridge(inertia=1.0, dissipation=-1.0, elasticity=10.0)
        with self.assertRaises(ValueError):
            IsomorphicPhysicsBridge(inertia=1.0, dissipation=0.1, elasticity=0.0)


class TestPQCZKSwarmConsensus(unittest.TestCase):

    def test_pqc_polynomial_ops(self):
        a = np.array([10, 20, 30], dtype=np.int64)
        b = np.array([5, 15, 25], dtype=np.int64)
        self.assertTrue(np.array_equal(poly_add(a, b), [15, 35, 55]))
        self.assertTrue(np.array_equal(poly_sub(a, b), [5, 5, 5]))

        cbd = sample_centered_binomial(eta=2, seed=b"test_seed_1234", nonce=1)
        self.assertEqual(len(cbd), 256)

    def test_pqc_ring_lwe_key_encapsulation(self):
        keypair = RingLWEPQC.generate_keypair()
        self.assertEqual(len(keypair.serialize_pubkey()), 32)

        ciphertext, shared_secret_sender = RingLWEPQC.encapsulate(keypair.public_key, keypair.seed_a)
        shared_secret_receiver = RingLWEPQC.decapsulate(keypair.secret_key, ciphertext)

        self.assertEqual(len(shared_secret_sender), 32)
        self.assertEqual(shared_secret_sender, shared_secret_receiver)

    def test_pqc_signatures(self):
        signer = PQCPostQuantumSigner()
        msg = b"MISSION_TARGET_COORD_WAYPOINT"
        sig_data = signer.sign_message(msg)
        self.assertTrue(signer.verify_signature(msg, sig_data))
        self.assertFalse(signer.verify_signature(b"CORRUPTED_PAYLOAD", sig_data))

    def test_zk_telemetry_prover_verifier_and_edge_cases(self):
        proof = ZKTelemetryProver.generate_proof(
            uav_id="UAV_007",
            uav_position=(10.0, 20.0, 30.0),
            uav_velocity=(1.0, 0.0, 0.0),
            target_estimate=(15.0, 22.0, 30.0),
            max_allowed_radius=50.0,
        )
        self.assertIn("uav_id", proof.to_dict())
        self.assertTrue(ZKTelemetryVerifier.verify_proof(proof))

        # Replay / Skew error
        future_t = time.time_ns() + 100_000_000_000
        self.assertFalse(ZKTelemetryVerifier.verify_proof(proof, current_time_ns=future_t))

        # Tampered challenge
        tampered_proof = ZKTelemetryProof(
            uav_id=proof.uav_id,
            commitment_pos=proof.commitment_pos,
            ephemeral_comm=proof.ephemeral_comm,
            claimed_radius_m=proof.claimed_radius_m,
            timestamp_ns=proof.timestamp_ns,
            challenge="deadbeef" * 8,
            response_proof=proof.response_proof,
        )
        self.assertFalse(ZKTelemetryVerifier.verify_proof(tampered_proof))

        # Invalid hex strings (ValueError)
        bad_hex_proof = ZKTelemetryProof(
            uav_id=proof.uav_id,
            commitment_pos="not_valid_hex",
            ephemeral_comm=proof.ephemeral_comm,
            claimed_radius_m=proof.claimed_radius_m,
            timestamp_ns=proof.timestamp_ns,
            challenge=proof.challenge,
            response_proof=proof.response_proof,
        )
        bad_hex_proof.challenge = hashlib.sha256(f"{bad_hex_proof.uav_id}:{bad_hex_proof.ephemeral_comm}:{bad_hex_proof.commitment_pos}:{bad_hex_proof.claimed_radius_m}:{bad_hex_proof.timestamp_ns}".encode()).hexdigest()
        self.assertFalse(ZKTelemetryVerifier.verify_proof(bad_hex_proof))

        # Out-of-group range Y_val = 0
        bad_y_proof = ZKTelemetryProof(
            uav_id=proof.uav_id,
            commitment_pos="0",
            ephemeral_comm=proof.ephemeral_comm,
            claimed_radius_m=proof.claimed_radius_m,
            timestamp_ns=proof.timestamp_ns,
            challenge=proof.challenge,
            response_proof=proof.response_proof,
        )
        bad_y_proof.challenge = hashlib.sha256(f"{bad_y_proof.uav_id}:{bad_y_proof.ephemeral_comm}:{bad_y_proof.commitment_pos}:{bad_y_proof.claimed_radius_m}:{bad_y_proof.timestamp_ns}".encode()).hexdigest()
        self.assertFalse(ZKTelemetryVerifier.verify_proof(bad_y_proof))

        # Out-of-range s_val >= ZK_ORDER
        bad_s_proof = ZKTelemetryProof(
            uav_id=proof.uav_id,
            commitment_pos=proof.commitment_pos,
            ephemeral_comm=proof.ephemeral_comm,
            claimed_radius_m=proof.claimed_radius_m,
            timestamp_ns=proof.timestamp_ns,
            challenge=proof.challenge,
            response_proof=hex(ZK_ORDER + 10)[2:],
        )
        bad_s_proof.challenge = hashlib.sha256(f"{bad_s_proof.uav_id}:{bad_s_proof.ephemeral_comm}:{bad_s_proof.commitment_pos}:{bad_s_proof.claimed_radius_m}:{bad_s_proof.timestamp_ns}".encode()).hexdigest()
        self.assertFalse(ZKTelemetryVerifier.verify_proof(bad_s_proof))

        # Out of bounds
        with self.assertRaises(ValueError):
            ZKTelemetryProver.generate_proof(
                uav_id="UAV_007",
                uav_position=(0.0, 0.0, 0.0),
                uav_velocity=(0.0, 0.0, 0.0),
                target_estimate=(1000.0, 1000.0, 1000.0),
                max_allowed_radius=10.0,
            )

    def test_bft_consensus_branches_and_tampered_blocks(self):
        nodes = [SwarmNode(f"drone_{i}", total_nodes=4, is_byzantine=(i == 3)) for i in range(4)]
        leader = nodes[0]
        block = leader.propose_target_detection(
            uav_pos=(100.0, 100.0, 20.0),
            uav_vel=(2.0, 0.0, 0.0),
            target_pos=(105.0, 102.0, 20.0),
            sensor_range_m=50.0,
        )

        # Byzantine node rejection
        self.assertIsNone(nodes[3].receive_pre_prepare(block))

        # Corrupted proof rejection
        corrupted_block = ConsensusBlock(
            block_index=1,
            view_number=0,
            proposer_id="drone_0",
            telemetry_proof=ZKTelemetryProof("bad", "0", "0", 10.0, 0, "bad", "0"),
            previous_block_hash="0000",
        )
        corrupted_block.block_hash = corrupted_block.compute_hash()
        self.assertIsNone(nodes[1].receive_pre_prepare(corrupted_block))

        # Corrupted hash rejection
        tampered_hash_block = ConsensusBlock(
            block_index=1,
            view_number=0,
            proposer_id=leader.node_id,
            telemetry_proof=block.telemetry_proof,
            previous_block_hash="0000",
            block_hash="bad_hash_1234",
        )
        self.assertIsNone(nodes[1].receive_pre_prepare(tampered_hash_block))

        # Pre-prepare
        prepares = []
        for n in nodes:
            res = n.receive_pre_prepare(block)
            if res:
                prepares.append(res)
        self.assertEqual(len(prepares), 3)

        # Invalid state process_prepare_vote check
        idle_node = SwarmNode("idle_node", 4)
        self.assertIsNone(idle_node.process_prepare_vote(prepares[0]))

        # Prepare
        commits = []
        for p in prepares:
            for n in nodes:
                c = n.process_prepare_vote(p)
                if c and c not in commits:
                    commits.append(c)
        self.assertEqual(len(commits), 3)

        # Commit
        for c in commits:
            for n in nodes:
                n.process_commit_vote(c)

        for i in range(3):
            self.assertEqual(len(nodes[i].blockchain), 1)
            self.assertEqual(nodes[i].state, ConsensusState.FINALIZED)

        # Brownout recovery
        nodes[3].recover_from_brownout(leader.blockchain)
        self.assertEqual(len(nodes[3].blockchain), 1)


class TestSiliconSideChannelGuard(unittest.TestCase):

    def test_constant_time_ops(self):
        self.assertEqual(ConstantTimeOps.ct_is_zero(0), 1)
        self.assertEqual(ConstantTimeOps.ct_is_zero(42), 0)
        self.assertEqual(ConstantTimeOps.ct_eq(1234, 1234), 1)
        self.assertEqual(ConstantTimeOps.ct_eq(1234, 4321), 0)
        self.assertEqual(ConstantTimeOps.ct_select(1, 0xAA, 0x55), 0xAA)
        self.assertEqual(ConstantTimeOps.ct_select(0, 0xAA, 0x55), 0x55)

        self.assertEqual(ConstantTimeOps.ct_select_bytes(1, b"TRUE", b"FLSE"), b"TRUE")
        self.assertEqual(ConstantTimeOps.ct_select_bytes(0, b"TRUE", b"FLSE"), b"FLSE")
        with self.assertRaises(ValueError):
            ConstantTimeOps.ct_select_bytes(1, b"SHORT", b"LONGER_BYTES")

        self.assertTrue(ConstantTimeOps.ct_memcmp(b"SECRET_KEY_12345", b"SECRET_KEY_12345"))
        self.assertFalse(ConstantTimeOps.ct_memcmp(b"SECRET_KEY_12345", b"WRONG_KEY_12345!"))
        self.assertFalse(ConstantTimeOps.ct_memcmp(b"A", b"AA"))

    def test_constant_time_aes(self):
        key = b"YELLOW SUBMARINE"
        pt = b"TOP_SECRET_UAV_X"
        ct = ConstantTimeAES.encrypt_block(pt, key)
        self.assertEqual(len(ct), 16)
        self.assertNotEqual(ct, pt)

        ct2 = ConstantTimeAES.encrypt_block(pt, key)
        self.assertEqual(ct, ct2)

        with self.assertRaises(ValueError):
            ConstantTimeAES.key_expansion(b"SHORT_KEY")
        with self.assertRaises(ValueError):
            ConstantTimeAES.encrypt_block(b"SHORT_PT", key)

    def test_constant_time_ring_arithmetic(self):
        a = 1500
        a_mont = a * ConstantTimeRingArithmetic.MONT_R
        red = ConstantTimeRingArithmetic.montgomery_reduce(a_mont)
        self.assertEqual(red % ConstantTimeRingArithmetic.Q, a)

        barr = ConstantTimeRingArithmetic.barrett_reduce(5000)
        self.assertEqual(barr, 5000 % 3329)

        poly_a = np.array([1, 2, 3], dtype=np.int64)
        poly_b = np.array([4, 5, 6], dtype=np.int64)
        sa, sb = ConstantTimeRingArithmetic.ct_cswap(poly_a.copy(), poly_b.copy(), 1)
        self.assertTrue(np.array_equal(sa, poly_b))
        self.assertTrue(np.array_equal(sb, poly_a))

        sa0, sb0 = ConstantTimeRingArithmetic.ct_cswap(poly_a.copy(), poly_b.copy(), 0)
        self.assertTrue(np.array_equal(sa0, poly_a))
        self.assertTrue(np.array_equal(sb0, poly_b))

    def test_clock_jitter_engine(self):
        jitter = ClockJitterEngine(min_jitter_cycles=5, max_jitter_cycles=20)
        cycles = jitter.inject_jitter()
        self.assertGreaterEqual(cycles, 5)
        self.assertLessEqual(cycles, 20)

        res = jitter.execute_with_jitter(lambda x, y: x + y, 10, 20)
        self.assertEqual(res, 30)

    def test_voltage_glitch_detector(self):
        detector = VoltageGlitchDetector(glitch_window_size=5)
        t0 = time.time_ns()
        self.assertTrue(detector.monitor_sample(3.3, t0))
        self.assertFalse(detector.monitor_sample(2.5, t0 + 1_000_000))
        self.assertFalse(detector.monitor_sample(3.3, t0 + 1_000_010))

        # Fill window to trigger eviction
        for i in range(10):
            detector.monitor_sample(3.3, t0 + 10_000_000 + i * 1_000_000)
        self.assertEqual(len(detector.samples), 5)

    def test_hardware_security_guard_and_zeroization(self):
        key = b"1234567890123456"
        guard = HardwareSecurityGuard(key)
        self.assertEqual(guard.state, SecurityState.SECURE)
        self.assertFalse(guard.is_compromised)

        ct = guard.secure_aes_encrypt(b"TELEMETRY_DATA_1")
        self.assertEqual(len(ct), 16)

        guard.process_voltage_telemetry(2.0)
        self.assertEqual(guard.state, SecurityState.SUSPICIOUS)
        guard.process_voltage_telemetry(2.0)
        guard.process_voltage_telemetry(2.0)

        self.assertEqual(guard.state, SecurityState.ZEROIZED)
        self.assertTrue(guard.is_compromised)
        self.assertEqual(guard._secure_key_store, bytearray(16))

        with self.assertRaises(PermissionError):
            guard.secure_aes_encrypt(b"TELEMETRY_DATA_1")

        with self.assertRaises(ValueError):
            HardwareSecurityGuard(b"SHORT")


class TestSEURadHardFirmware(unittest.TestCase):

    def test_hamming_sec_ded(self):
        for n in range(16):
            c = HammingSECDED.encode_nibble(n)
            rec, status = HammingSECDED.decode_nibble(c)
            self.assertEqual(rec, n)
            self.assertEqual(status, ECCStatus.NO_ERROR)

            for bit in range(8):
                c_corrupted = c ^ (1 << bit)
                rec_corr, st_corr = HammingSECDED.decode_nibble(c_corrupted)
                self.assertEqual(rec_corr, n)
                self.assertEqual(st_corr, ECCStatus.SINGLE_ERROR_CORRECTED)

            c_double = c ^ (1 << 2) ^ (1 << 4)
            _, st_double = HammingSECDED.decode_nibble(c_double)
            self.assertEqual(st_double, ECCStatus.DOUBLE_ERROR_DETECTED)

        # Byte codec direct tests with single & double error
        clean_enc = bytearray(HammingSECDED.encode_bytes(b"TEST"))
        clean_enc[0] ^= (1 << 2)  # single error in lo
        clean_enc[1] ^= (1 << 3)  # single error in hi
        rec_b, st_b, n_corr = HammingSECDED.decode_and_correct_bytes(bytes(clean_enc))
        self.assertEqual(rec_b, b"TEST")
        self.assertEqual(st_b, ECCStatus.SINGLE_ERROR_CORRECTED)

        clean_enc[2] ^= (1 << 1)
        clean_enc[2] ^= (1 << 5)  # double error
        _, st_double_b, _ = HammingSECDED.decode_and_correct_bytes(bytes(clean_enc))
        self.assertEqual(st_double_b, ECCStatus.DOUBLE_ERROR_DETECTED)
        # Byte codec error check
        with self.assertRaises(ValueError):
            HammingSECDED.decode_and_correct_bytes(b"ODD_3")

    def test_memory_scrubber_in_place_repair_and_double_error(self):
        payload = b"CRITICAL_AVIONICS_FLIGHT_PLAN_COORDS"
        scrubber = MemoryScrubber(payload)

        # Inject single bit flip
        scrubber.inject_bit_flip(byte_index=4, bit_index=3)
        scrubber.inject_bit_flip(byte_index=5, bit_index=2)
        res = scrubber.scrub_cycle()
        self.assertEqual(res["corrections_made"], 2)

        clean_data, status = scrubber.read_clean_data()
        self.assertEqual(clean_data, payload)
        self.assertEqual(status, ECCStatus.NO_ERROR)

        # Inject double bit flip in byte 0 and byte 1
        scrubber.inject_bit_flip(byte_index=0, bit_index=1)
        scrubber.inject_bit_flip(byte_index=0, bit_index=5)
        scrubber.inject_bit_flip(byte_index=1, bit_index=2)
        scrubber.inject_bit_flip(byte_index=1, bit_index=6)
        res_double = scrubber.scrub_cycle()
        self.assertGreater(res_double["uncorrectable_errors"], 0)

        with self.assertRaises(IndexError):
            scrubber.inject_bit_flip(9999, 0)
        with self.assertRaises(ValueError):
            scrubber.inject_bit_flip(0, 9)

    def test_quaternion_and_tmr_voting(self):
        # Zero norm fallback
        q_zero = Quaternion.from_array(np.array([0.0, 0.0, 0.0, 0.0]))
        self.assertEqual(q_zero.w, 1.0)

        # Quaternion distance clamping
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(q1.distance_to(q2), 0.0)

        tmr_default = TripleModularAttitudeVoting()
        self.assertEqual(tmr_default.reg_a.w, 1.0)
        tmr = TripleModularAttitudeVoting(q1)
        tmr.update_state(Quaternion(1.0, 0.0, 0.0, 0.0))

        # All 3 match
        consensus, healthy = tmr.vote_and_repair()
        self.assertTrue(healthy)

        # Corrupt Reg C
        tmr.inject_seu_flip("c", "x", 0.8)
        consensus, repaired = tmr.vote_and_repair()
        self.assertTrue(repaired)
        self.assertEqual(tmr.seu_recoveries_count, 1)

        # Corrupt Reg A
        tmr.inject_seu_flip("a", "y", 0.9)
        consensus_a, repaired_a = tmr.vote_and_repair()
        self.assertTrue(repaired_a)

        # Corrupt Reg B
        tmr.inject_seu_flip("b", "z", 0.9)
        consensus_b, repaired_b = tmr.vote_and_repair()
        self.assertTrue(repaired_b)

        # Critical divergence (all 3 corrupted to differ)
        tmr.reg_a = Quaternion(1.0, 0.0, 0.0, 0.0)
        tmr.reg_b = Quaternion(0.0, 1.0, 0.0, 0.0)
        tmr.reg_c = Quaternion(0.0, 0.0, 1.0, 0.0)
        _, success = tmr.vote_and_repair()
        self.assertFalse(success)

    def test_overcurrent_crowbar_controller(self):
        crowbar = OvercurrentCrowbarController(nominal_current_a=1.5, trip_threshold_a=3.5, hard_surge_threshold_a=6.0)
        self.assertEqual(crowbar.state, CrowbarState.ARMED)

        # Normal current
        self.assertEqual(crowbar.evaluate_current(1.2), CrowbarState.ARMED)

        # Hard surge trip
        t0 = time.time()
        self.assertEqual(crowbar.evaluate_current(7.0, current_time_s=t0), CrowbarState.TRIPPED)
        self.assertEqual(crowbar.trip_history[-1]["type"], "HARD_SURGE")
        self.assertEqual(crowbar.evaluate_current(2.0, current_time_s=t0 + 0.1), CrowbarState.TRIPPED)

        # Cooldown period elapsed
        self.assertEqual(crowbar.evaluate_current(2.0, current_time_s=t0 + 0.6), CrowbarState.COOLING_DOWN)
        # Drops to nominal
        self.assertEqual(crowbar.evaluate_current(1.0, current_time_s=t0 + 0.7), CrowbarState.ARMED)

        # Manual reset
        crowbar.evaluate_current(4.0)
        self.assertEqual(crowbar.trip_history[-1]["type"], "OVERCURRENT_TRIP")
        self.assertTrue(crowbar.reset_latch())
        self.assertEqual(crowbar.state, CrowbarState.ARMED)


class TestGeoAcousticSubterraneanSLAM(unittest.TestCase):

    def test_crustal_magnetic_map_and_gradient(self):
        cmap = CrustalMagneticMap(base_field_nt=np.array([20000.0, 5000.0, 45000.0]))
        pos = np.array([10.0, 20.0, -50.0])
        field = cmap.sample_field(pos)
        self.assertEqual(len(field), 3)

        grad = cmap.compute_gradient_tensor(pos)
        self.assertEqual(grad.shape, (3, 3))

    def test_ultrasonic_phased_array_beamforming(self):
        array = UltrasonicPhasedArray(num_channels=8)
        targets = [(0.5, 3.0, 0.0), (-1.0, 4.5, 0.0)]
        signals = array.simulate_echo_signals(targets_local=targets, max_range_m=6.0, noise_std=0.01)
        self.assertEqual(signals.shape[0], 8)

        angles = np.linspace(-45.0, 45.0, 19)
        r_grid, a_grid, intensity = array.delay_and_sum_beamform(signals, angles, max_range_m=6.0)
        self.assertEqual(intensity.shape, (19, signals.shape[1]))

        points = array.extract_point_cloud(intensity, r_grid, a_grid, threshold=0.5)
        self.assertIsInstance(points, list)

    def test_subterranean_ekf_slam_pipeline(self):
        cmap = CrustalMagneticMap(base_field_nt=np.array([20000.0, 5000.0, 45000.0]))
        init_state = np.array([0.0, 0.0, -10.0, 1.0, 0.0, 0.0])
        ekf = SubterraneanEKFSLAM(init_state, cmap)

        # Predict
        ekf.predict(dt=0.1, accel_body=np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(ekf.state[0], 0.1)

        # Update with magnetic observation
        measured_b = cmap.sample_field(np.array([0.1, 0.0, -10.0])) + np.array([1.0, -0.5, 0.2])
        ekf.update_magnetic_anomaly(measured_b)

        # Update with ultrasonic points
        ekf.update_ultrasonic_points([(0.5, 2.0, 0.0)])
        self.assertEqual(len(ekf.map_points), 1)

        with self.assertRaises(ValueError):
            SubterraneanEKFSLAM(np.array([1.0, 2.0]), cmap)


class TestBioMycoDroneAirframe(unittest.TestCase):

    def test_material_and_airframe_fem(self):
        mat = MycoChitinMaterial()
        frame = MycoDroneFEMAirframe(arm_length_m=0.25, arm_cross_section_radius_m=0.012, material=mat)

        deflection_data = frame.compute_arm_tip_deflection(motor_thrust_n=12.5)
        self.assertGreater(deflection_data["tip_deflection_mm"], 0.0)
        self.assertGreater(deflection_data["safety_factor"], 1.0)
        self.assertIn("total_frame_mass_g", deflection_data)

    def test_modal_frequencies_and_acoustic_damping(self):
        frame = MycoDroneFEMAirframe()
        freqs = frame.compute_modal_frequencies(num_modes=3)
        self.assertEqual(len(freqs), 3)
        self.assertLess(freqs[0], freqs[1])
        self.assertLess(freqs[1], freqs[2])

        f_spectrum = np.array([50.0, 100.0, 500.0, 1000.0, 2000.0])
        tl = frame.compute_acoustic_transmission_loss(f_spectrum)
        self.assertEqual(len(tl), len(f_spectrum))
        self.assertTrue(np.all(tl >= 0.0))

    def test_biodegradation_simulator(self):
        sim_res = BiodegradationSimulator.simulate_mass_loss(
            initial_mass_g=250.0,
            days=90,
            temperature_c=25.0,
            soil_relative_humidity=0.80,
            base_half_life_days=45.0,
        )
        self.assertEqual(len(sim_res["time_days"]), 91)
        self.assertAlmostEqual(sim_res["remaining_mass_g"][45], 125.0, delta=5.0)
        self.assertAlmostEqual(sim_res["mass_loss_percent"][45], 50.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
