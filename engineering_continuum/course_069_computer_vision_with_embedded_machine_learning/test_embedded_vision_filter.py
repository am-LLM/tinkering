import numpy as np
from embedded_vision_filter import EmbeddedSobelFilter

def test_embedded_sobel_edge_detection():
    img = np.zeros((10, 10), dtype=np.int32)
    img[:, 5:] = 200
    filtered = EmbeddedSobelFilter.convolve2d(img)
    assert filtered.shape == (8, 8)
    assert np.max(filtered) > 100
