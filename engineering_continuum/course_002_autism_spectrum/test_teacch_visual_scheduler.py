from teacch_visual_scheduler import TEACCHScheduler, TaskState
def test_teacch_progression():
    sched = TEACCHScheduler(["Sensory Calm", "Math Block", "Break"])
    task = sched.start_next_task()
    assert task["task"] == "Sensory Calm"
    assert sched.complete_current_task() is True
