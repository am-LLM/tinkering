class OLCAssembler:
    def __init__(self, reads: list, min_overlap: int = 3):
        self.reads = reads
        self.min_overlap = min_overlap

    def find_overlap(self, a: str, b: str) -> int:
        max_ov = 0
        for l in range(self.min_overlap, min(len(a), len(b)) + 1):
            if a.endswith(b[:l]):
                max_ov = l
        return max_ov

    def greedy_assemble(self) -> str:
        seqs = list(self.reads)
        while len(seqs) > 1:
            best_pair = (0, 1)
            best_ov = -1
            for i in range(len(seqs)):
                for j in range(len(seqs)):
                    if i != j:
                        ov = self.find_overlap(seqs[i], seqs[j])
                        if ov > best_ov:
                            best_ov = ov
                            best_pair = (i, j)
            if best_ov < self.min_overlap:
                break
            i, j = best_pair
            merged = seqs[i] + seqs[j][best_ov:]
            seqs.pop(max(i, j))
            seqs.pop(min(i, j))
            seqs.append(merged)
        return seqs[0]
