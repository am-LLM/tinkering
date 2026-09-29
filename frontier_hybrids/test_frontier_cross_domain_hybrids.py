"""
Unit Tests for Frontier Cross-Domain Hybrid Projects
Asserts real physics, Markov chains, Randles impedance, chaos embedding, and PQC zero-knowledge proofs.
"""

import pytest
import numpy as np
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from quantum_attribution_beacon import QuantumOpticalBeacon, MetaMarketingMarkovAttributionEngine
from hydro_acoustic_wash_mesh import HydroAcousticBiofilmSolver
from polyvagal_supply_chain_damper import PolyvagalSupplyChainDamper
from bio_optic_lingual_plasmonic import SurfacePlasmonResonanceSolver
from pqc_zk_infrastructure_sentry import SubthresholdMosfetLeakageDetector, PostQuantumZeroKnowledgeSentry

def test_quantum_optical_beacon_coherence():
    beacon = QuantumOpticalBeacon()
    g2 = beacon.calculate_second_order_coherence(dark_count_rate=30.0, coincidence_rate=1500.0)
    assert g2 < 0.5, f"Expected quantum single photon anti-bunching (g2 < 0.5), got {g2}"
    assert beacon.verify_quantum_authenticity(g2) is True

def test_meta_marketing_markov_attribution():
    emitters = ["JAMMER_ALPHA", "JAMMER_BETA", "JAMMER_GAMMA"]
    engine = MetaMarketingMarkovAttributionEngine(emitters)
    
    # Journeys of packet exposures
    journeys = [
        ["JAMMER_ALPHA", "JAMMER_BETA", "PACKET_DROP"],
        ["JAMMER_ALPHA", "SUCCESS"],
        ["JAMMER_BETA", "JAMMER_GAMMA", "PACKET_DROP"],
        ["JAMMER_ALPHA", "JAMMER_BETA", "PACKET_DROP"],
        ["JAMMER_GAMMA", "SUCCESS"]
    ]
    trans_matrix = engine.build_transition_matrix(journeys)
    assert trans_matrix.shape == (len(engine.states), len(engine.states))
    assert np.allclose(trans_matrix.sum(axis=1), 1.0)
    
    effects = engine.calculate_removal_effects(trans_matrix)
    assert len(effects) == 3
    assert abs(sum(effects.values()) - 1.0) < 1e-4

def test_hydro_acoustic_wash_solver():
    solver = HydroAcousticBiofilmSolver()
    f_shed = solver.calculate_vortex_shedding_frequency(superficial_velocity_m_s=0.4)
    assert 300.0 < f_shed < 600.0, f"Expected shed frequency in 300-600 Hz, got {f_shed}"
    
    z_real, z_imag = solver.calculate_randles_biofilm_impedance(biofilm_thickness_microns=50.0, frequency_hz=f_shed)
    assert z_real > 50.0
    assert z_imag < 0.0  # Capacitive / Warburg imaginary component
    
    health = solver.evaluate_filter_health(viv_frequency_hz=f_shed, z_real_ohms=z_real)
    assert "clogging_index" in health
    assert "water_potability_pct" in health

def test_polyvagal_supply_chain_damper():
    damper = PolyvagalSupplyChainDamper(embedding_dimension=3, time_delay_tau=2)
    
    # Synthetic diverging panic orders (exponential oscillations)
    t = np.linspace(0, 10, 30)
    panic_orders = 100.0 + 20.0 * np.exp(0.15 * t) * np.sin(2.0 * np.pi * 0.5 * t)
    
    attractor = damper.reconstruct_phase_space(panic_orders)
    assert attractor.shape[1] == 3
    
    lyapunov = damper.calculate_lyapunov_divergence(attractor)
    order, metrics = damper.compute_vagal_counter_pacing(
        current_demand=150.0,
        lyapunov_exponent=lyapunov,
        retail_inventory=80.0,
        target_inventory=120.0
    )
    assert order > 0.0
    assert "variance_reduction_pct" in metrics

def test_bio_optic_spr_solver():
    spr = SurfacePlasmonResonanceSolver()
    theta = spr.calculate_spr_resonance_angle(ambient_refractive_index=1.0003)
    assert 30.0 < theta < 60.0
    
    eval_result = spr.evaluate_breath_metabolic_distress(
        baseline_refractive_index=1.0003,
        acetone_ppm=2.5,
        ammonia_ppm=1.2
    )
    assert eval_result["metabolic_crisis_alert"] is True
    assert eval_result["distress_severity_score"] > 0.0

def test_pqc_zk_infrastructure_sentry():
    mosfet = SubthresholdMosfetLeakageDetector()
    i_sub = mosfet.calculate_leakage_current(v_gs=0.2, v_ds=1.2)
    assert i_sub > 0.0
    
    tamper, z_score = mosfet.detect_firmware_tampering(current_leakage_ua=i_sub * 2.5, baseline_leakage_ua=i_sub)
    assert tamper is True
    
    sentry = PostQuantumZeroKnowledgeSentry("GRID_NODE_TEST")
    proof = sentry.generate_zk_proof(tamper, z_score)
    assert sentry.verify_zk_proof(proof) is True
    assert proof["jidoka_airgap_tripped"] is True
