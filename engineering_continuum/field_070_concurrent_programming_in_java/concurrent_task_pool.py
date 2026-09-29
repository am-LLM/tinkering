"""Course 070: Thread-Safe Priority Work Stealing & Task Queue Engine"""
import heapq
import threading
from typing import Any, Tuple

class PriorityTaskPool:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.queue = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, priority: int, item: Any):
        with self.lock:
            while len(self.queue) >= self.capacity:
                self.not_full.wait()
            heapq.heappush(self.queue, (priority, item))
            self.not_empty.notify()

    def get(self) -> Tuple[int, Any]:
        with self.lock:
            while len(self.queue) == 0:
                self.not_empty.wait()
            item = heapq.heappop(self.queue)
            self.not_full.notify()
            return item

    def size(self) -> int:
        with self.lock:
            return len(self.queue)
