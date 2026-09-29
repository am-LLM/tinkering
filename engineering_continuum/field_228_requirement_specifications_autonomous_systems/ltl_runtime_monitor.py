"""Course 228: Linear Temporal Logic (LTL) Runtime Trace Safety Monitor"""
class LTLRuntimeMonitor:
    @staticmethod
    def check_globally_p(trace: list, predicate_fn) -> bool:
        # G(p): predicate must hold at every time step
        return all(predicate_fn(state) for state in trace)

    @staticmethod
    def check_finally_p(trace: list, predicate_fn) -> bool:
        # F(p): predicate must hold at least once
        return any(predicate_fn(state) for state in trace)

    @staticmethod
    def check_p_until_q(trace: list, pred_p, pred_q) -> bool:
        # p U q: p holds until q occurs
        for state in trace:
            if pred_q(state):
                return True
            if not pred_p(state):
                return False
        return False
