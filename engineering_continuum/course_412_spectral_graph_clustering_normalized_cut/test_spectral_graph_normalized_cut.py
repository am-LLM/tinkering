from spectral_graph_normalized_cut import SpectralGraphNormalizedCut
import numpy as np

def test_spectralgraphnormalizedcut_transform():
    obj = SpectralGraphNormalizedCut(parameter=1.2, length=5)
    out = obj.execute_transform(np.linspace(0, 1, 5))
    assert len(out) == 5
    assert obj.verify_properties()
    assert obj.get_norm() > 0.0

def test_spectralgraphnormalizedcut_default():
    obj = SpectralGraphNormalizedCut(parameter=0.5, length=3)
    out = obj.execute_transform()
    assert len(out) == 3
    assert np.allclose(out, np.cos(0.5) * 0.5)
