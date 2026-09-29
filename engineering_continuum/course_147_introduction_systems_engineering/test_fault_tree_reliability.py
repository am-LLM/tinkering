from fault_tree_reliability import FaultTreeNode

def test_fta():
    e1 = FaultTreeNode('BASIC', prob=0.1)
    e2 = FaultTreeNode('BASIC', prob=0.2)
    and_gate = FaultTreeNode('AND', [e1, e2])
    assert abs(and_gate.evaluate_unreliability() - 0.02) < 1e-4
    or_gate = FaultTreeNode('OR', [e1, e2])
    assert abs(or_gate.evaluate_unreliability() - 0.28) < 1e-4
