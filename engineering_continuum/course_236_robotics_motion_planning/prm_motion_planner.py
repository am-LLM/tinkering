"""Course 236: Probabilistic Roadmap (PRM) Graph Roadmap Query Pathfinder"""
import heapq
import numpy as np

class PRMMotionPlanner:
    def __init__(self):
        self.roadmap = {}

    def add_roadmap_edge(self, u: tuple, v: tuple):
        d = np.hypot(u[0]-v[0], u[1]-v[1])
        self.roadmap.setdefault(u, []).append((v, d))
        self.roadmap.setdefault(v, []).append((u, d))

    def query(self, start: tuple, goal: tuple) -> float:
        dist = {start: 0.0}
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                return d
            if d > dist.get(u, float('inf')):
                continue
            for v, cost in self.roadmap.get(u, []):
                if d + cost < dist.get(v, float('inf')):
                    dist[v] = d + cost
                    heapq.heappush(pq, (dist[v], v))
        return float('inf')
