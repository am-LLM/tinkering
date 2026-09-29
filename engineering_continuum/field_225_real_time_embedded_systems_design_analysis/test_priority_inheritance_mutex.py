from priority_inheritance_mutex import PIPMutex

def test_pip_mutex():
    mutex = PIPMutex("shared_res")
    mutex.acquire("TaskLow", 1)
    inherited = mutex.acquire("TaskHigh", 10)
    assert inherited == 10 # Elevated
    restored = mutex.release()
    assert restored == 1
