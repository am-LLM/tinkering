import numpy as np
from int8_quantization_engine import INT8Quantizer

def test_quantization():
    w = np.linspace(-1.0, 1.0, 10)
    qw, scale, zp = INT8Quantizer.quantize(w)
    deq = INT8Quantizer.dequantize(qw, scale, zp)
    assert np.max(np.abs(w - deq)) < 0.05
