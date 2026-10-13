import pytest
import numpy as np
from engineering_continuum.field_423_tdoa_rf_emitter_geolocator.rf_tdoa_geolocator import TDoARFEmitterGeolocator

def test_field_423_initialization():
    geo = TDoARFEmitterGeolocator()
    assert geo.c == 299792458.0
    assert geo.receiver_nodes.shape == (3, 3)

def test_field_423_rf_emitter_triangulation():
    geo = TDoARFEmitterGeolocator()
    
    # Ground truth transmitter location at (800, 500)
    tx_pos = np.array([800.0, 500.0])
    s0 = geo.receiver_nodes[0][:2]
    s1 = geo.receiver_nodes[1][:2]
    s2 = geo.receiver_nodes[2][:2]
    
    d0 = np.linalg.norm(tx_pos - s0)
    d1 = np.linalg.norm(tx_pos - s1)
    d2 = np.linalg.norm(tx_pos - s2)
    
    tau_10 = (d1 - d0) / geo.c
    tau_20 = (d2 - d0) / geo.c
    
    est = geo.compute_emitter_2d_fix([tau_10, tau_20])
    assert np.linalg.norm(est - tx_pos) < 100.0 # Within grid discretization
    assert bool(geo.verify_properties()) is True
