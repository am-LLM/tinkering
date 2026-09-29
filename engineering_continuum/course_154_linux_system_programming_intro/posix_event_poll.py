"""Course 154: POSIX Event Multiplexer & Non-Blocking Stream Poll Simulator"""
class PosixEventPoll:
    EPOLLIN = 1
    EPOLLOUT = 2

    def __init__(self):
        self.registered_fds = {}

    def register(self, fd: int, events: int):
        self.registered_fds[fd] = events

    def poll(self, ready_readable: list, ready_writable: list) -> list:
        events = []
        for fd, mask in self.registered_fds.items():
            ev = 0
            if (mask & self.EPOLLIN) and fd in ready_readable:
                ev |= self.EPOLLIN
            if (mask & self.EPOLLOUT) and fd in ready_writable:
                ev |= self.EPOLLOUT
            if ev > 0:
                events.append((fd, ev))
        return events
