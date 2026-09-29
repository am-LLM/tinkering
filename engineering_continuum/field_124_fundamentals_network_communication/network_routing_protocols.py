"""Course 124: Dijkstra Shortest Path Link-State Routing Algorithm"""
import heapq

class LinkStateRouter:
    def __init__(self):
        self.graph = {}

    def add_link(self, u: str, v: str, cost: float):
        self.graph.setdefault(u, []).append((v, cost))
        self.graph.setdefault(v, []).append((u, cost))

    def shortest_path(self, source: str, destination: str) -> tuple:
        dist = {source: 0.0}
        prev = {}
        pq = [(0.0, source)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if u == destination:
                break
            if d > dist.get(u, float('inf')):
                continue
            for v, cost in self.graph.get(u, []):
                if dist.get(u, float('inf')) + cost < dist.get(v, float('inf')):
                    dist[v] = dist[u] + cost
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
                    
        path = []
        curr = destination
        while curr in prev:
            path.append(curr)
            curr = prev[curr]
        if curr == source:
            path.append(source)
            path.reverse()
        return dist.get(destination, float('inf')), path
