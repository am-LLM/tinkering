"""Course 225: Priority Inheritance Protocol (PIP) Mutex Inversion Resolver"""
class PIPMutex:
    def __init__(self, name: str):
        self.name = name
        self.holder = None

    def acquire(self, task_id: str, task_priority: int) -> int:
        if self.holder is None:
            self.holder = {"id": task_id, "base_prio": task_priority, "inherited_prio": task_priority}
            return task_priority
        else:
            # Priority inheritance: elevate holder priority to max
            if task_priority > self.holder["inherited_prio"]:
                self.holder["inherited_prio"] = task_priority
            return self.holder["inherited_prio"]

    def release(self) -> int:
        if self.holder:
            base = self.holder["base_prio"]
            self.holder = None
            return base
        return 0
