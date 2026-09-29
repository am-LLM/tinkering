"""Course 130: De Bruijn Graph k-mer Assembly & Eulerian Path Finder"""
class DeBruijnAssembler:
    def __init__(self, k: int = 3):
        self.k = k
        self.graph = {}

    def build_graph(self, reads: list):
        for r in reads:
            for i in range(len(r) - self.k + 1):
                kmer = r[i:i+self.k]
                prefix = kmer[:-1]
                suffix = kmer[1:]
                self.graph.setdefault(prefix, []).append(suffix)

    def assemble_simple_path(self, start_node: str) -> str:
        curr = start_node
        res = curr
        while curr in self.graph and self.graph[curr]:
            next_node = self.graph[curr].pop(0)
            res += next_node[-1]
            curr = next_node
        return res
