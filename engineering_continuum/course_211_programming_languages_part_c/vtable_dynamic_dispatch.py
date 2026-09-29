"""Course 211: Virtual Method Table (VTable) Dynamic Dispatch & Method Resolution"""
class VTableClass:
    def __init__(self, name: str, parent=None):
        self.name = name
        self.vtable = parent.vtable.copy() if parent else {}

    def add_method(self, method_name: str, func):
        self.vtable[method_name] = func

    def dispatch(self, instance, method_name: str, *args):
        if method_name not in self.vtable:
            raise AttributeError(f"Method {method_name} not found in {self.name} vtable")
        return self.vtable[method_name](instance, *args)
