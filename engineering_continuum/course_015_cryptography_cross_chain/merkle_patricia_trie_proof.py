import hashlib

def sha256(val: str) -> str:
    return hashlib.sha256(val.encode('utf-8')).hexdigest()

class MerkleTree:
    def __init__(self, leaves: list):
        self.leaves = [sha256(l) for l in leaves]
        self.tree = [self.leaves]
        self._build()

    def _build(self):
        current = self.leaves
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                l = current[i]
                r = current[i+1] if i+1 < len(current) else l
                next_level.append(sha256(l + r))
            self.tree.append(next_level)
            current = next_level

    def get_root(self) -> str:
        return self.tree[-1][0] if self.tree else ""
