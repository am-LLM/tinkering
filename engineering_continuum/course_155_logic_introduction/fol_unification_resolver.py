"""Course 155: First-Order Logic Robinson Unification Algorithm"""
class FOLUnification:
    @classmethod
    def unify(cls, term1, term2, subst: dict = None) -> dict:
        if subst is None:
            subst = {}
        if term1 == term2:
            return subst
        elif isinstance(term1, str) and term1.startswith("?"):
            return cls._unify_var(term1, term2, subst)
        elif isinstance(term2, str) and term2.startswith("?"):
            return cls._unify_var(term2, term1, subst)
        elif isinstance(term1, list) and isinstance(term2, list):
            if len(term1) != len(term2):
                return None
            for t1, t2 in zip(term1, term2):
                subst = cls.unify(t1, t2, subst)
                if subst is None:
                    return None
            return subst
        return None

    @classmethod
    def _unify_var(cls, var: str, x, subst: dict) -> dict:
        if var in subst:
            return cls.unify(subst[var], x, subst)
        subst_copy = subst.copy()
        subst_copy[var] = x
        return subst_copy
