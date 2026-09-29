"""
Unit Tests for RF Anti-Jamming DSP: Adaptive IIR Notch Filter & Spatial Null-Steering Beamformer
"""

import pytest
import numpy as np
from rf_anti_jam_dsp import (
    AdaptiveIIRNotchFilter, CascadedAdaptiveNotchFilter,
    SpatialNullSteeringBeamformer, ULAArrayConfig
)


def test_iir_notch_filter_attenuation():
    fs = 10000.0
    f_notch = 2000.0
    filter_iir = AdaptiveIIRNotchFilter(fs=fs, rho=0.98, initial_f0=f_notch, adaptive=False)
    
    test_freqs = np.array([f_notch, 500.0])
    H = filter_iir.get_frequency_response(test_freqs)
    
    notch_gain_db = 20.0 * np.log10(np.abs(H[0]))
    passband_gain_db = 20.0 * np.log10(np.abs(H[1]))
    
    assert notch_gain_db < -35.0, f"Notch attenuation insufficient: {notch_gain_db} dB"
    assert np.isclose(passband_gain_db, 0.0, atol=1.0), f"Passband distorted: {passband_gain_db} dB"


def test_iir_notch_filter_default_init():
    # initial_f0=None branch (defaults to a = 0.0 / fs/4)
    filt = AdaptiveIIRNotchFilter(fs=8000.0, initial_f0=None)
    assert np.isclose(filt.notch_frequency, 2000.0)


def test_adaptive_notch_jammer_tracking():
    fs = 10000.0
    t = np.arange(4000) / fs
    f_jammer = 1500.0
    
    clean_soi = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    jammer = 10.0 * np.sin(2 * np.pi * f_jammer * t)
    noisy_rx = clean_soi + jammer
    
    filter_iir = AdaptiveIIRNotchFilter(fs=fs, rho=0.95, mu=0.05, initial_f0=2500.0, adaptive=True)
    filtered = filter_iir.filter_signal(noisy_rx)
    
    estimated_f = filter_iir.notch_frequency
    assert np.isclose(estimated_f, f_jammer, atol=25.0), f"Frequency tracking failed: {estimated_f} vs {f_jammer}"
    
    tail_input_pwr = np.mean(noisy_rx[-1000:] ** 2)
    tail_output_pwr = np.mean(filtered[-1000:] ** 2)
    rejection_ratio = tail_input_pwr / max(tail_output_pwr, 1e-12)
    assert rejection_ratio > 10.0, f"Jammer power not suppressed: {rejection_ratio}"


def test_cascaded_multi_tone_notch():
    fs = 10000.0
    t = np.arange(3000) / fs
    j1 = 5.0 * np.sin(2 * np.pi * 1000.0 * t)
    j2 = 5.0 * np.sin(2 * np.pi * 3000.0 * t)
    soi = 1.0 * np.sin(2 * np.pi * 300.0 * t)
    sig = j1 + j2 + soi
    
    cascaded = CascadedAdaptiveNotchFilter(num_notches=2, fs=fs, initial_f0s=[1000.0, 3000.0], rho=0.98, adaptive=False)
    out = cascaded.filter_signal(sig)
    
    in_pwr = np.mean(sig ** 2)
    out_pwr = np.mean(out[-1000:] ** 2)
    assert out_pwr < 0.1 * in_pwr, f"Cascaded rejection failed: in={in_pwr}, out={out_pwr}"


def test_spatial_beamformer_mvdr_null_steering():
    num_elements = 8
    beamformer = SpatialNullSteeringBeamformer(ULAArrayConfig(num_elements=num_elements))
    
    theta_soi = 0.0
    theta_j1 = 30.0
    theta_j2 = -45.0
    
    num_samples = 1000
    t = np.arange(num_samples)
    
    s_soi = np.exp(1j * 2 * np.pi * 0.05 * t)
    s_j1 = 20.0 * np.exp(1j * 2 * np.pi * 0.12 * t)
    s_j2 = 20.0 * np.exp(1j * 2 * np.pi * 0.18 * t)
    
    a_soi = beamformer.steering_vector(theta_soi)[:, np.newaxis]
    a_j1 = beamformer.steering_vector(theta_j1)[:, np.newaxis]
    a_j2 = beamformer.steering_vector(theta_j2)[:, np.newaxis]
    noise = (np.random.randn(num_elements, num_samples) + 1j * np.random.randn(num_elements, num_samples)) * 0.1
    
    X = a_soi @ s_soi[np.newaxis, :] + a_j1 @ s_j1[np.newaxis, :] + a_j2 @ s_j2[np.newaxis, :] + noise
    
    weights = beamformer.compute_mvdr_weights(X, target_soi_deg=theta_soi)
    
    response_soi = np.abs(np.vdot(weights, beamformer.steering_vector(theta_soi)))
    gain_soi_db = 20 * np.log10(response_soi)
    assert np.isclose(gain_soi_db, 0.0, atol=0.5), f"SOI gain not unity: {gain_soi_db} dB"
    
    response_j1 = np.abs(np.vdot(weights, beamformer.steering_vector(theta_j1)))
    gain_j1_db = 20 * np.log10(max(response_j1, 1e-12))
    assert gain_j1_db < -20.0, f"Jammer 1 not nulled: {gain_j1_db} dB"
    
    response_j2 = np.abs(np.vdot(weights, beamformer.steering_vector(theta_j2)))
    gain_j2_db = 20 * np.log10(max(response_j2, 1e-12))
    assert gain_j2_db < -20.0, f"Jammer 2 not nulled: {gain_j2_db} dB"

    # Test beamforming output
    y = beamformer.beamform(X)
    assert len(y) == num_samples
    
    # Test beampattern computation
    angles = np.linspace(-90, 90, 181)
    beampattern = beamformer.compute_beampattern(angles)
    assert len(beampattern) == len(angles)


def test_spatial_beamformer_lcmv():
    beamformer = SpatialNullSteeringBeamformer(ULAArrayConfig(num_elements=8))
    angles = [10.0, -30.0]
    responses = [1.0 + 0j, 0.0 + 0j]
    
    X = (np.random.randn(8, 200) + 1j * np.random.randn(8, 200))
    w = beamformer.compute_lcmv_weights(X, angles, responses)
    
    g1 = np.abs(np.vdot(w, beamformer.steering_vector(10.0)))
    g2 = np.abs(np.vdot(w, beamformer.steering_vector(-30.0)))
    
    assert np.isclose(g1, 1.0, atol=1e-3)
    assert np.isclose(g2, 0.0, atol=1e-3)
