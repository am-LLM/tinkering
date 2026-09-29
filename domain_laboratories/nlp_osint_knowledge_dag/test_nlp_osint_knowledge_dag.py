"""
Comprehensive Unit Test Suite for NLP, OSINT, Comparative Eschatology & Bastion Kernel.

Covers:
1. Socio-Cognitive Shield (VAD Extraction, Hostility Index, Crisis De-escalation MDP).
2. Humor & OSINT Shield (DSP Micro-Acoustic Prosody, NLP Sarcasm & Deception Defense).
3. Polymath Eschatology DAG (Topological Sorting, Cross-Corpus Affinity, Geopolitical Correlator).
4. Antigravity Omni-Node (SMT Invariant Gating, Multi-Channel Priority Dispatcher, Cryptographic Attestation).
"""

import math
import numpy as np
import pytest
import time

from socio_cognitive_shield import (
    HostilityState,
    TacticalEmpathyAction,
    SentimentValenceVector,
    HostilityMetrics,
    SocioCognitiveAnalyzer,
    CrisisDeescalationMDP,
    TacticalEmpathyGenerator,
)

from humor_osint_shield import (
    MicroAcousticFeatures,
    NLPDeceptionScores,
    DeceptionAssessment,
    MicroAcousticProsodyExtractor,
    NLPSarcasmDeceptionAnalyzer,
    HumorOSINTShield,
)

from polymath_eschatology_dag import (
    Tradition,
    EschatologicalArchetype,
    EschatologyNode,
    GeopoliticalEvent,
    AlignmentReport,
    EschatologyDAGEngine,
    CrossTraditionSemanticAffinity,
    GeopoliticalEschatologyCorrelator,
)

from antigravity_omni_node import (
    PriorityChannel,
    NodeOperationalState,
    KernelEvent,
    NodeRuntimeInvariants,
    HeartbeatAttestation,
    FormalInvariantGate,
    AntigravityOmniNode,
)


# ============================================================================
# 1. SOCIO-COGNITIVE SHIELD TESTS
# ============================================================================

class TestSocioCognitiveShield:
    """Test suite for Sentiment Valence, Linguistic Hostility & Crisis De-escalation MDP."""

    def test_vad_sentiment_extraction(self):
        analyzer = SocioCognitiveAnalyzer()

        # Calm, positive text
        calm_text = "Thank you so much, I really appreciate your wonderful help and calm support."
        vad_calm = analyzer.extract_vad(calm_text)
        assert vad_calm.valence > 0.3
        assert vad_calm.arousal < 0.2

        # Highly agitated, hostile text
        hostile_text = "YOU IDIOT! FIX THIS DISASTER RIGHT NOW OR I WILL DESTROY YOU!!!"
        vad_hostile = analyzer.extract_vad(hostile_text)
        assert vad_hostile.valence < -0.3
        assert vad_hostile.arousal > 0.3
        assert vad_hostile.dominance > 0.2

    def test_hostility_index_and_state_classification(self):
        analyzer = SocioCognitiveAnalyzer()

        # Test Mild/Calm text
        m_calm = analyzer.analyze_hostility("Hello, could we please review the recent status update?")
        assert m_calm.state in (HostilityState.CALM, HostilityState.MILD_TENSION)
        assert m_calm.hostility_index < 0.30

        # Test Critical/Explosive text
        m_crit = analyzer.analyze_hostility("YOU STUPID MORONS! DO YOUR JOB IMMEDIATELY AND SHUT UP!! AS I SAID, YOU ARE USELESS CLOWNS!!!")
        assert m_crit.state in (HostilityState.ACUTE_HOSTILITY, HostilityState.CRITICAL_EXPLOSIVE)
        assert m_crit.hostility_index > 0.60
        assert m_crit.profanity_density > 0.0
        assert m_crit.casing_punctuation_agitation > 0.0

    def test_crisis_deescalation_mdp_and_value_iteration(self):
        mdp = CrisisDeescalationMDP(gamma=0.95)

        # Value function must be strictly monotonically decreasing with higher hostility
        # (Calm state has highest value / lowest penalty)
        assert mdp.V_star[0] > mdp.V_star[1] > mdp.V_star[2] > mdp.V_star[3] > mdp.V_star[4]

        # Optimal policy lookup
        opt_action_calm = mdp.get_optimal_action(HostilityState.CALM)
        assert isinstance(opt_action_calm, TacticalEmpathyAction)

        opt_action_crit = mdp.get_optimal_action(HostilityState.CRITICAL_EXPLOSIVE)
        assert opt_action_crit in (TacticalEmpathyAction.STRATEGIC_SILENCE, TacticalEmpathyAction.LABEL_EMOTION)

    def test_tactical_empathy_generation_and_convergence(self):
        generator = TacticalEmpathyGenerator()
        hostile_input = "We are completely blocked and running out of time!"

        result = generator.generate_response(hostile_input)
        assert 'metrics' in result
        assert 'optimal_action' in result
        assert len(result['response_text']) > 0

        # Simulation: Trajectory from CRITICAL state converges to CALM
        trajectory = generator.simulate_conversation_deescalation(
            initial_hostility_state=HostilityState.CRITICAL_EXPLOSIVE,
            max_turns=12,
            seed=42
        )
        assert len(trajectory) > 0
        final_state = trajectory[-1]['state_after']
        assert final_state in ("CALM", "MILD_TENSION", "ELEVATED_AGITATION")


# ============================================================================
# 2. HUMOR & OSINT SHIELD TESTS
# ============================================================================

class TestHumorOSINTShield:
    """Test suite for Micro-Acoustic Prosody & Multimodal Deception Detection."""

    def test_micro_acoustic_prosody_extraction(self):
        extractor = MicroAcousticProsodyExtractor(sample_rate_hz=16000)

        # Synthesize audio with target F0=150Hz, Jitter=2%, Shimmer=5%
        synthetic_speech = MicroAcousticProsodyExtractor.synthesize_mock_speech(
            duration_s=1.0, f0_hz=150.0, jitter_amp=0.02, shimmer_amp=0.05, seed=123
        )
        features = extractor.extract_features(synthetic_speech)

        assert pytest.approx(features.f0_mean_hz, rel=0.15) == 150.0
        assert features.pitch_jitter_pct > 0.0
        assert features.spectral_shimmer_pct > 0.0
        assert features.spectral_centroid_hz > 500.0
        assert features.speaking_cadence_hz > 0.0

    def test_nlp_sarcasm_and_social_engineering_scoring(self):
        analyzer = NLPSarcasmDeceptionAnalyzer()

        # Sarcastic text with valence contrast
        sarcastic_text = "Oh brilliant, what a wonderful disaster and terrible mess."
        scores_sarcastic = analyzer.score_text(sarcastic_text)
        assert scores_sarcastic.sarcasm_score > 0.35

        # Social engineering spear-phishing attack text
        soc_eng_text = "URGENT: By direct order of the CEO and CISO, immediately execute wire transfer before deadline."
        scores_soc_eng = analyzer.score_text(soc_eng_text)
        assert scores_soc_eng.urgency_pressure_score > 0.40
        assert scores_soc_eng.authority_gradient_score > 0.30

    def test_multimodal_deception_fusion(self):
        shield = HumorOSINTShield(sample_rate_hz=16000)

        # 1. Genuine communication
        gen_text = "Here is the verified weekly report. Let me know if you need any additional figures."
        gen_audio = MicroAcousticProsodyExtractor.synthesize_mock_speech(
            duration_s=1.0, f0_hz=130.0, jitter_amp=0.01, shimmer_amp=0.02, seed=42
        )
        assessment_gen = shield.evaluate_communication(gen_text, gen_audio)
        assert assessment_gen.is_deceptive is False
        assert assessment_gen.social_engineering_risk_level in ("LOW", "MODERATE")

        # 2. High-threat social engineering communication
        threat_text = "CONFIDENTIAL: CISO and Board require immediate emergency action to prevent network suspension now!"
        threat_audio = MicroAcousticProsodyExtractor.synthesize_mock_speech(
            duration_s=1.0, f0_hz=220.0, jitter_amp=0.06, shimmer_amp=0.12, seed=99
        )
        assessment_threat = shield.evaluate_communication(threat_text, threat_audio)
        assert assessment_threat.is_deceptive is True
        assert assessment_threat.composite_deception_score > assessment_gen.composite_deception_score


# ============================================================================
# 3. POLYMATH ESCHATOLOGY DAG TESTS
# ============================================================================

class TestPolymathEschatologyDAG:
    """Test suite for Eschatology DAG ontology, Topological Sorting & Geopolitical Correlation."""

    def test_dag_acyclicity_and_topological_sort(self):
        dag = EschatologyDAGEngine()

        assert dag.is_acyclic() is True
        topo_order = dag.topological_sort()
        assert len(topo_order) == len(dag.nodes)

        # Assert Islamic causal ordering in topological sort
        assert topo_order.index("ISL_01") < topo_order.index("ISL_02")
        assert topo_order.index("ISL_02") < topo_order.index("ISL_03")
        assert topo_order.index("ISL_03") < topo_order.index("ISL_04")
        assert topo_order.index("ISL_04") < topo_order.index("ISL_05")
        assert topo_order.index("ISL_05") < topo_order.index("ISL_06")

        # Assert Biblical causal ordering
        assert topo_order.index("BIB_01") < topo_order.index("BIB_02")
        assert topo_order.index("BIB_02") < topo_order.index("BIB_03")
        assert topo_order.index("BIB_03") < topo_order.index("BIB_04")

    def test_cross_tradition_semantic_affinity(self):
        dag = EschatologyDAGEngine()
        affinity_calc = CrossTraditionSemanticAffinity(dag)
        matrix = affinity_calc.compute_archetypal_affinity_matrix()

        assert matrix[(Tradition.ISLAMIC, Tradition.ISLAMIC)] == 1.0
        assert matrix[(Tradition.BIBLICAL, Tradition.BIBLICAL)] == 1.0

        # Islamic and Biblical share major archetypes (Deceiver, Deliverer, Tribulation, Battle, Restoration)
        isl_bib_affinity = matrix[(Tradition.ISLAMIC, Tradition.BIBLICAL)]
        assert isl_bib_affinity >= 0.35

        # Vedic and Rabbinic share Moral Decay and Golden Age archetypes
        ved_rab_affinity = matrix[(Tradition.VEDIC, Tradition.RABBINIC)]
        assert ved_rab_affinity >= 0.25

    def test_geopolitical_event_correlation(self):
        correlator = GeopoliticalEschatologyCorrelator()

        # Real-world event: Euphrates river drought
        event = GeopoliticalEvent(
            event_id="GEO_001",
            timestamp_iso="2026-09-24T00:00:00Z",
            location="Middle East / Iraq / Syria",
            summary="Satellite radar confirms unprecedented drought and river water depletion along the Euphrates river basin uncovering dry bed.",
            keywords={"euphrates", "river", "drought", "water", "iraq", "syria"},
            severity_score=0.85
        )

        report = correlator.correlate_event(event)
        assert report.event_id == "GEO_001"
        assert len(report.top_matching_nodes) > 0
        assert report.top_matching_nodes[0][0] == "ISL_01"  # Matches Drying of Euphrates
        assert report.weaponization_risk_index > 0.40


# ============================================================================
# 4. ANTIGRAVITY OMNI-NODE BASTION KERNEL TESTS
# ============================================================================

class TestAntigravityOmniNode:
    """Test suite for Formal SMT Invariant Gating, Prioritized Dispatcher & Cryptographic Attestation."""

    def test_prioritized_event_dispatching(self):
        node = AntigravityOmniNode(node_id="TEST-NODE-01")
        processed_events: List[str] = []

        def generic_handler(ev: KernelEvent):
            processed_events.append(ev.event_id)

        for ch in PriorityChannel:
            node.register_handler(ch, generic_handler)

        # Enqueue Normal priority (P3) first, then Critical priority (P0)
        sig_crit = node.gate.sign_payload("EV_CRIT", str(sorted({"alert": "shutdown"}.items())))
        node.submit_event("EV_NORMAL_1", PriorityChannel.NORMAL, "USER", {"msg": "hello"})
        node.submit_event("EV_NORMAL_2", PriorityChannel.NORMAL, "USER", {"msg": "world"})
        node.submit_event("EV_CRIT", PriorityChannel.CRITICAL, "SYSTEM_GUARD", {"alert": "shutdown"}, signature=sig_crit)

        # Process next event -> Critical (P0) MUST be dequeued before Normal (P3)
        ev1 = node.process_next_event()
        assert ev1 is not None
        assert ev1.event_id == "EV_CRIT"
        assert ev1.channel == PriorityChannel.CRITICAL

        ev2 = node.process_next_event()
        assert ev2 is not None
        assert ev2.channel == PriorityChannel.NORMAL

    def test_formal_invariant_signature_and_quarantine(self):
        node = AntigravityOmniNode(node_id="TEST-NODE-02")

        # Submit privileged SYSTEM event WITHOUT valid signature
        accepted, reason = node.submit_event(
            "EV_UNAUTHORIZED",
            PriorityChannel.SYSTEM,
            "ATTACKER",
            {"cmd": "reflash_firmware"},
            signature="bogus_signature"
        )

        assert accepted is False
        assert "INVARIANT_VIOLATION" in reason
        # Node must instantly transition to QUARANTINED state upon privileged invariant rupture
        assert node.state == NodeOperationalState.QUARANTINED

        # Under QUARANTINED state, regular NORMAL events are rejected by invariant gate
        acc_norm, r_norm = node.submit_event("EV_ROUTINE", PriorityChannel.NORMAL, "USER", {"data": 123})
        assert acc_norm is False
        assert "QUARANTINED" in r_norm

    def test_cryptographic_heartbeat_merkle_attestation(self):
        node = AntigravityOmniNode(node_id="TEST-NODE-03")

        # Generate consecutive heartbeats
        hb1 = node.generate_heartbeat_attestation()
        hb2 = node.generate_heartbeat_attestation()
        hb3 = node.generate_heartbeat_attestation()

        assert hb1.sequence_index == 1
        assert hb2.sequence_index == 2
        assert hb3.sequence_index == 3

        # Merkle chaining: hb2's prev_root == hb1's current_merkle_root
        assert hb2.prev_merkle_root == hb1.current_merkle_root
        assert hb3.prev_merkle_root == hb2.current_merkle_root

        # Verify unbroken chain integrity
        assert node.verify_heartbeat_chain() is True

        # Tamper with an intermediate heartbeat record
        node.heartbeat_chain[1].current_merkle_root = "corrupted_hash"
        assert node.verify_heartbeat_chain() is False

    def test_high_throughput_saturation_benchmark(self):
        node = AntigravityOmniNode(node_id="BENCH-NODE")
        n_events = 10000

        t0 = time.perf_counter()
        # Ingest 10,000 events
        for i in range(n_events):
            node.submit_event(f"BENCH_{i}", PriorityChannel.TELEMETRY, "SENSOR", {"val": i})

        t_ingest = time.perf_counter() - t0
        assert t_ingest < 3.0  # Ingest rate > 3,300 - 15,000 events/s

        # Process 10,000 events
        t1 = time.perf_counter()
        processed = 0
        while True:
            ev = node.process_next_event()
            if ev is None:
                break
            processed += 1

        t_process = time.perf_counter() - t1
        assert processed == n_events
        assert t_process < 1.5
