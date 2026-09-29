class AhoCorasickAutomaton:
    def __init__(self, patterns: list):
        self.patterns = patterns
        self.trie = {}
        self._build_trie()

    def _build_trie(self):
        for p in self.patterns:
            curr = self.trie
            for char in p:
                if char not in curr:
                    curr[char] = {}
                curr = curr[char]
            curr['#'] = p

    def search_text(self, text: str) -> list:
        matches = []
        for i in range(len(text)):
            curr = self.trie
            for j in range(i, len(text)):
                if text[j] in curr:
                    curr = curr[text[j]]
                    if '#' in curr:
                        matches.append((i, curr['#']))
                else:
                    break
        return matches
