import numpy as np
from cnn_feature_extractor import CNNFeatureExtractor

def test_cnn_conv_pool():
    x = np.random.randn(8, 8).astype(np.float32)
    k = np.ones((3, 3), dtype=np.float32)
    conv_out = CNNFeatureExtractor.conv2d(x, k)
    assert conv_out.shape == (6, 6)
