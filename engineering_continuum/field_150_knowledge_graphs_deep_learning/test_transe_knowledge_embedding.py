import numpy as np
from transe_knowledge_embedding import TransEEmbedding

def test_transe():
    h = np.array([1.0, 0.0])
    r = np.array([0.0, 1.0])
    t = np.array([1.0, 1.0])
    score = TransEEmbedding.score_triple(h, r, t)
    assert abs(score) < 1e-5
