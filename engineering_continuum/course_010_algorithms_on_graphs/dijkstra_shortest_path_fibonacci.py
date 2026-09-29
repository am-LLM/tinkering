import heapq

def dijkstra_shortest_path(graph: dict, start_node: str) -> dict:
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    pq = [(0, start_node)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > distances[u]:
            continue
        for v, weight in graph[u].items():
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                heapq.heappush(pq, (distances[v], v))
    return distances
