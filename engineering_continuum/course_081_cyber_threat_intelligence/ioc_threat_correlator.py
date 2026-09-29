"""Course 081: STIX/TAXII Threat Indicator of Compromise (IOC) Graph Correlator"""
from typing import Dict, List, Set

class IOCThreatCorrelator:
    def __init__(self):
        self.graph = {}
        self.ioc_types = {}

    def register_ioc(self, ioc: str, ioc_type: str):
        self.ioc_types[ioc] = ioc_type
        if ioc not in self.graph:
            self.graph[ioc] = set()

    def add_correlation(self, ioc_a: str, ioc_b: str):
        self.graph.setdefault(ioc_a, set()).add(ioc_b)
        self.graph.setdefault(ioc_b, set()).add(ioc_a)

    def find_campaign_cluster(self, start_ioc: str) -> Set[str]:
        visited = set()
        queue = [start_ioc]
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                queue.extend(self.graph.get(curr, set()) - visited)
        return visited
