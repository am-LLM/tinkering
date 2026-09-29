"""Course 178: Jukes-Cantor & Kimura 2-Parameter Evolutionary Nucleotide Distance"""
import math

class JukesCantorEvolution:
    @staticmethod
    def jukes_cantor_distance(seq1: str, seq2: str) -> float:
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        p = differences / len(seq1) if seq1 else 0.0
        if p >= 0.75:
            return float('inf')
        return float(-0.75 * math.log(1.0 - (4.0 / 3.0) * p))
