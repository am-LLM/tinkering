"""Course 132: Burrows-Wheeler Transform (BWT) & Suffix Search Engine"""
class FMIndexBWTSearch:
    @staticmethod
    def bwt_transform(s: str) -> str:
        s_term = s + "$"
        rotations = sorted([s_term[i:] + s_term[:i] for i in range(len(s_term))])
        return "".join(r[-1] for r in rotations)

    @staticmethod
    def count_occurrences(text: str, pattern: str) -> int:
        count = 0
        idx = 0
        while True:
            idx = text.find(pattern, idx)
            if idx == -1:
                break
            count += 1
            idx += 1
        return count
