"""Course 147: Fault Tree Analysis (FTA) Minimal Cut Set & Unreliability Evaluator"""
class FaultTreeNode:
    def __init__(self, node_type: str, children: list = None, prob: float = 0.0):
        self.type = node_type # 'AND', 'OR', 'BASIC'
        self.children = children or []
        self.prob = prob

    def evaluate_unreliability(self) -> float:
        if self.type == 'BASIC':
            return self.prob
        elif self.type == 'AND':
            p = 1.0
            for c in self.children:
                p *= c.evaluate_unreliability()
            return float(p)
        elif self.type == 'OR':
            p_comp = 1.0
            for c in self.children:
                p_comp *= (1.0 - c.evaluate_unreliability())
            return float(1.0 - p_comp)
        return 0.0
