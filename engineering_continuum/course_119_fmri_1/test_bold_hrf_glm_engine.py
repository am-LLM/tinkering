import numpy as np
from bold_hrf_glm_engine import BOLDHRFGLMEngine

def test_bold_hrf():
    stim = np.zeros(50)
    stim[10] = 1.0
    resp = BOLDHRFGLMEngine.convolve_stimulus(stim)
    assert len(resp) == 50
    assert np.max(resp) > 0.0
