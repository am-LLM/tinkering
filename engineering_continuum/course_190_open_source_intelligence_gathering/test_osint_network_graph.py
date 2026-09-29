from osint_network_graph import OSINTNetworkGraph

def test_osint_graph():
    g = OSINTNetworkGraph()
    g.add_edge("TargetA", "Alias1")
    g.add_edge("TargetA", "Email1")
    cent = g.degree_centrality()
    assert cent["TargetA"] == 1.0
