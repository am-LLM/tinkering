import pytest
import numpy as np
from engineering_continuum.field_422_crpa_mvdr_beamforming.crpa_mvdr_beamforming import CRPAMVDRBeamformer

def test_field_422_initialization():
    beamformer = CRPAMVDRBeamformer(num_elements=4)
    assert beamformer.num_elements == 4
    assert len(beamformer.element_positions) == 4

def test_field_422_steering_vector():
    beamformer = CRPAMVDRBeamformer(num_elements=4)
    a_boresight = beamformer.compute_steering_vector(0.0)
    assert np.allclose(a_boresight, np.ones(4, dtype=complex))

def test_field_422_jammer_null_steering():
    beamformer = CRPAMVDRBeamformer(num_elements=4)
    
    # Target satellite overhead at 0 deg, Ground jammer at 40 deg
    weights = beamformer.synthesize_mvdr_weights(
        target_satellite_az_deg=0.0,
        jamming_signal_az_deg=[40.0],
        jamming_snr_db=30.0
    )
    
    sat_gain = beamformer.compute_array_gain_db(weights, 0.0)
    jam_gain = beamformer.compute_array_gain_db(weights, 40.0)
    
    # Null depth must suppress jammer by > 25 dB
    assert (sat_gain - jam_gain) > 25.0
    assert bool(beamformer.verify_properties()) is True
