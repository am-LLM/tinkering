from vtable_dynamic_dispatch import VTableClass

def test_vtable_dispatch():
    base = VTableClass("Base")
    base.add_method("speak", lambda self: "Base speaking")
    derived = VTableClass("Derived", parent=base)
    derived.add_method("speak", lambda self: "Derived speaking")
    
    assert base.dispatch(None, "speak") == "Base speaking"
    assert derived.dispatch(None, "speak") == "Derived speaking"
