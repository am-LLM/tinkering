import numpy as np
from engine_44_holographic_data_storage_bragg_grating import HolographicStorageBraggEngine

def test_holographic_storage():
    engine = HolographicStorageBraggEngine(page_dim=8, seed=42)
    page1 = np.ones((8, 8))
    page2 = np.zeros((8, 8))
    engine.store_data_page(30.0, page1)
    engine.store_data_page(35.0, page2)

    res = engine.readout_page(30.0)
    assert res["diffraction_efficiency"] > 0.5
    assert res["target_angle_matched_deg"] == 30.0
