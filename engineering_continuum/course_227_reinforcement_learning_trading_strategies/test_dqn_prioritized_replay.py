from dqn_prioritized_replay import PrioritizedReplayBuffer

def test_per_buffer():
    buf = PrioritizedReplayBuffer(10)
    for i in range(5):
        buf.add((i, i+1), priority=(i+1.0))
    sample = buf.sample(2)
    assert len(sample) == 2
