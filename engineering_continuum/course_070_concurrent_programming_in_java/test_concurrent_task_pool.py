from concurrent_task_pool import PriorityTaskPool

def test_priority_task_pool():
    pool = PriorityTaskPool(capacity=5)
    pool.put(10, "low")
    pool.put(1, "high")
    pool.put(5, "med")
    assert pool.size() == 3
    p1, item1 = pool.get()
    assert item1 == "high"
