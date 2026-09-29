"""Course 091: Raft Consensus State Machine & Term Log Replication"""
class RaftNode:
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"

    def __init__(self, node_id: int, peers: list):
        self.node_id = node_id
        self.peers = peers
        self.state = self.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []

    def start_election(self):
        self.state = self.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        votes = 1
        return votes

    def append_entries(self, leader_term: int, entries: list) -> bool:
        if leader_term < self.current_term:
            return False
        self.current_term = leader_term
        self.state = self.FOLLOWER
        self.log.extend(entries)
        return True
