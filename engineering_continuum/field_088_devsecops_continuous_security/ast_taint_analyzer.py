"""Course 088: Static AST Security Taint Vulnerability Analyzer"""
import ast

class TaintAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.tainted_vars = set()
        self.vulnerabilities = []

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') in ['input', 'request_get']:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tainted_vars.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        sink_names = ['eval', 'system', 'exec']
        func_name = getattr(node.func, 'id', '')
        if func_name in sink_names:
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in self.tainted_vars:
                    self.vulnerabilities.append((func_name, arg.id))
        self.generic_visit(node)
