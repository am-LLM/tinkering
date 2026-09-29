"""Course 209: Untyped Lambda Calculus AST & Beta-Reduction Evaluator"""
class LambdaTerm:
    def __init__(self, term_type: str, var: str = None, body = None, left = None, right = None):
        self.type = term_type # 'VAR', 'ABS', 'APP'
        self.var = var
        self.body = body
        self.left = left
        self.right = right

class LambdaEvaluator:
    @classmethod
    def eval_step(cls, term: LambdaTerm):
        if term.type == 'APP':
            if term.left.type == 'ABS':
                return cls.substitute(term.left.body, term.left.var, term.right)
        return term

    @classmethod
    def substitute(cls, body: LambdaTerm, var_name: str, replacement: LambdaTerm) -> LambdaTerm:
        if body.type == 'VAR' and body.var == var_name:
            return replacement
        return body
