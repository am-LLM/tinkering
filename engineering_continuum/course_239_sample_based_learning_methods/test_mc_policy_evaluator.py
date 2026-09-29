from mc_policy_evaluator import MCPolicyEvaluator

def test_mc_eval():
    mc = MCPolicyEvaluator()
    mc.update_first_visit(["S1", "S2", "S1"], [1.0, 2.0, 0.0], gamma=1.0)
    assert mc.v["S1"] == 3.0
    assert mc.v["S2"] == 2.0
