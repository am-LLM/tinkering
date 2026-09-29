from posix_event_poll import PosixEventPoll

def test_poll():
    ep = PosixEventPoll()
    ep.register(3, PosixEventPoll.EPOLLIN)
    ep.register(4, PosixEventPoll.EPOLLOUT)
    res = ep.poll(ready_readable=[3], ready_writable=[])
    assert res == [(3, PosixEventPoll.EPOLLIN)]
