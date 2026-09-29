from collections import deque

class MaxFlowNetwork:
    def __init__(self, num_nodes: int):
        self.n = num_nodes
        self.capacity = [[0] * num_nodes for _ in range(num_nodes)]

    def add_edge(self, u: int, v: int, cap: int):
        self.capacity[u][v] = cap

    def edmonds_karp(self, source: int, sink: int) -> int:
        parent = [-1] * self.n
        max_flow = 0
        while True:
            parent = [-1] * self.n
            parent[source] = -2
            q = deque([(source, float('inf'))])
            flow = 0
            while q:
                u, cur_flow = q.popleft()
                if u == sink:
                    flow = cur_flow
                    break
                for v in range(self.n):
                    if parent[v] == -1 and self.capacity[u][v] > 0:
                        parent[v] = u
                        q.append((v, min(cur_flow, self.capacity[u][v])))
            if flow == 0:
                break
            max_flow += flow
            curr = sink
            while curr != source:
                prev = parent[curr]
                self.capacity[prev][curr] -= flow
                self.capacity[curr][prev] += flow
                curr = prev
        return max_flow
