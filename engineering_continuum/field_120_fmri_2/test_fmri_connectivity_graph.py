import numpy as np
from fmri_connectivity_graph import FMRIConnectivityGraph

def test_fmri_graph():
    ts = np.random.randn(5, 100)
    corr = FMRIConnectivityGraph.correlation_matrix(ts)
    adj = FMRIConnectivityGraph.binarize_network(corr, threshold=0.1)
    deg = FMRIConnectivityGraph.degree_centrality(adj)
    assert len(deg) == 5
