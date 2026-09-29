"""Course 093: Transition/Transversion Ratio (Ts/Tv) & CpG Island Markov Detector"""
class DNAMutationAnalyzer:
    PURINES = {'A', 'G'}
    PYRIMIDINES = {'C', 'T'}

    @classmethod
    def calculate_ts_tv(cls, original: str, mutated: str) -> dict:
        transitions = 0
        transversions = 0
        for b1, b2 in zip(original.upper(), mutated.upper()):
            if b1 != b2:
                if (b1 in cls.PURINES and b2 in cls.PURINES) or (b1 in cls.PYRIMIDINES and b2 in cls.PYRIMIDINES):
                    transitions += 1
                else:
                    transversions += 1
        ratio = transitions / transversions if transversions > 0 else float('inf')
        return {"transitions": transitions, "transversions": transversions, "ts_tv_ratio": ratio}

    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        seq = sequence.upper()
        gc = seq.count('G') + seq.count('C')
        return gc / len(seq) if seq else 0.0
