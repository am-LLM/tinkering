import pytest
import numpy as np
from engineering_continuum.field_421_acoustic_aero_tdoa_triangulator.acoustic_tdoa_triangulator import AcousticAeroTDoATriangulator

def test_field_421_initialization():
    triangulator = AcousticAeroTDoATriangulator()
    assert triangulator.c == 343.0
    assert triangulator.target_harmonic == 110.0
    assert triangulator.sensor_locations.shape == (3, 3)

def test_field_421_acoustic_engine_harmonic_detection():
    triangulator = AcousticAeroTDoATriangulator(target_harmonic_hz=115.0)
    
    t = np.arange(8000) / 8000.0
    # Engine harmonic signature + low ambient background noise
    audio = np.sin(2 * np.pi * 115.0 * t) * 4.0 + np.random.normal(0, 0.2, 8000)
    
    detected, snr = triangulator.detect_engine_harmonic_signature(audio, sample_rate=8000.0)
    assert detected is True
    assert snr > 5.0

def test_field_421_tdoa_triangulation():
    triangulator = AcousticAeroTDoATriangulator()
    
    # Ground truth source at (600, 400)
    source = np.array([600.0, 400.0])
    s_a = triangulator.sensor_locations[0][:2]
    s_b = triangulator.sensor_locations[1][:2]
    s_c = triangulator.sensor_locations[2][:2]
    
    d_a = np.linalg.norm(source - s_a)
    d_b = np.linalg.norm(source - s_b)
    d_c = np.linalg.norm(source - s_c)
    
    tau_ba = (d_b - d_a) / triangulator.c
    tau_ca = (d_c - d_a) / triangulator.c
    
    est = triangulator.triangulate_2d_tdoa([tau_ba, tau_ca])
    # Estimate should be within search grid tolerance (~60 meters)
    assert np.linalg.norm(est - source) < 80.0
    assert bool(triangulator.verify_properties()) is True
