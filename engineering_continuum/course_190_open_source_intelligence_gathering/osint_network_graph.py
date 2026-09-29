"""Course 190: OSINT Entity Relationship Bipartite Network & Degree Centrality Graph"""
class OSINTNetworkGraph:
    def __init__(self):
        self.nodes = set()
        self.adj = {}

    def add_edge(self, u: str, v: str):
        self.nodes.add(u)
        self.nodes.add(v)
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    def degree_centrality(self) -> dict:
        n = len(self.nodes)
        if n <= 1:
            return {node: 0.0 for node in self.nodes}
        return {node: len(self.adj.get(node, set())) / (n - 1) for node in self.nodes}
