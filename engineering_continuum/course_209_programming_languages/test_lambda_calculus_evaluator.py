from lambda_calculus_evaluator import LambdaTerm, LambdaEvaluator

def test_lambda_eval():
    # ((lambda x. x) y) -> y
    id_func = LambdaTerm('ABS', var='x', body=LambdaTerm('VAR', var='x'))
    app = LambdaTerm('APP', left=id_func, right=LambdaTerm('VAR', var='y'))
    res = LambdaEvaluator.eval_step(app)
    assert res.type == 'VAR'
    assert res.var == 'y'
