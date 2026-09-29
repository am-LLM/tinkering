"""Course 092: Smith-Waterman Local Sequence Alignment Algorithm"""
import numpy as np

class SmithWatermanAligner:
    def __init__(self, match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -1):
        self.match = match_score
        self.mismatch = mismatch_penalty
        self.gap = gap_penalty

    def align(self, seq1: str, seq2: str) -> tuple:
        n, m = len(seq1), len(seq2)
        h = np.zeros((n + 1, m + 1), dtype=int)
        
        max_score = 0
        max_pos = (0, 0)
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score_match = h[i-1, j-1] + (self.match if seq1[i-1] == seq2[j-1] else self.mismatch)
                score_del = h[i-1, j] + self.gap
                score_ins = h[i, j-1] + self.gap
                h[i, j] = max(0, score_match, score_del, score_ins)
                if h[i, j] > max_score:
                    max_score = h[i, j]
                    max_pos = (i, j)
                    
        return int(max_score), max_pos
