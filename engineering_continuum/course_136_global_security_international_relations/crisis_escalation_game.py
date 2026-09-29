"""Course 136: Crisis Escalation Bargaining Tree & Deterrence Equilibrium"""
class CrisisEscalationGame:
    @staticmethod
    def subgame_perfect_outcome(challenger_cost_war: float, defender_resolve: float) -> str:
        if defender_resolve > challenger_cost_war:
            return "CHALLENGER_BACKS_DOWN"
        return "DEFENDER_CONCEDES"
