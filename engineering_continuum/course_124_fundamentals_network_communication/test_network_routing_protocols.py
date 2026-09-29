from network_routing_protocols import LinkStateRouter

def test_dijkstra_router():
    r = LinkStateRouter()
    r.add_link("A", "B", 1.0)
    r.add_link("B", "C", 2.0)
    r.add_link("A", "C", 5.0)
    d, path = r.shortest_path("A", "C")
    assert d == 3.0
    assert path == ["A", "B", "C"]
