from fol_unification_resolver import FOLUnification

def test_unification():
    t1 = ["P", "?x", "b"]
    t2 = ["P", "a", "?y"]
    s = FOLUnification.unify(t1, t2)
    assert s["?x"] == "a"
    assert s["?y"] == "b"
