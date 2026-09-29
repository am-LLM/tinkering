import ast
from ast_taint_analyzer import TaintAnalyzer

def test_taint():
    tree = ast.parse("y = input(); eval(y)")
    t = TaintAnalyzer()
    t.visit(tree)
    assert len(t.vulnerabilities) == 1
