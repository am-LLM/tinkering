from davis_putnam_sat_solver import DPLLSolver

def test_dpll_sat_solver():
    clauses = [[1, 2], [-1, 2], [1, -2]]
    is_sat, assignment = DPLLSolver.dpll(clauses)
    assert is_sat is True
    assert assignment[1] is True
    assert assignment[2] is True
