import numpy as np
from pagerank_graph_engine import PageRankGraphEngine

def test_pagerank():
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=float)
    pr = PageRankGraphEngine.compute_pagerank(adj)
    assert len(pr) == 3
    assert abs(np.sum(pr) - 1.0) < 1e-4
