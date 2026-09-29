"""Course 061: Resource Description Framework (RDF) & SPARQL Pattern Matcher"""
from typing import List, Tuple, Dict

class RDFGraphStore:
    def __init__(self):
        self.triples = set()

    def add_triple(self, s: str, p: str, o: str):
        self.triples.add((s, p, o))

    def query_pattern(self, pattern: Tuple[str, str, str]) -> List[Dict[str, str]]:
        s_pat, p_pat, o_pat = pattern
        results = []
        for s, p, o in self.triples:
            bindings = {}
            if s_pat.startswith("?"):
                bindings[s_pat] = s
            elif s_pat != s:
                continue
                
            if p_pat.startswith("?"):
                bindings[p_pat] = p
            elif p_pat != p:
                continue
                
            if o_pat.startswith("?"):
                bindings[o_pat] = o
            elif o_pat != o:
                continue
                
            results.append(bindings)
        return results
