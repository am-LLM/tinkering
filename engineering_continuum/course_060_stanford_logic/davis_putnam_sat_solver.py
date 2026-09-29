"""Course 060: Davis-Putnam-Logemann-Loveland (DPLL) Boolean SAT Solver"""
class DPLLSolver:
    @staticmethod
    def dpll(clauses: list, assignment: dict = None) -> tuple:
        if assignment is None:
            assignment = {}
            
        if any(len(c) == 0 for c in clauses):
            return False, {}
            
        if len(clauses) == 0:
            return True, assignment
            
        unit_clauses = [c for c in clauses if len(c) == 1]
        if unit_clauses:
            lit = unit_clauses[0][0]
            val = lit > 0
            var = abs(lit)
            new_assignment = assignment.copy()
            new_assignment[var] = val
            new_clauses = [
                [l for l in c if l != -lit]
                for c in clauses if lit not in c
            ]
            return DPLLSolver.dpll(new_clauses, new_assignment)
            
        lit = clauses[0][0]
        var = abs(lit)
        res_t, assign_t = DPLLSolver.dpll([[var]] + clauses, assignment.copy())
        if res_t:
            return True, assign_t
        return DPLLSolver.dpll([[-var]] + clauses, assignment.copy())
