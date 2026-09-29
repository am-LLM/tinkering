from raft_consensus_core import RaftNode

def test_raft_node():
    node = RaftNode(1, [2, 3])
    votes = node.start_election()
    assert votes == 1
    assert node.state == RaftNode.CANDIDATE
    success = node.append_entries(leader_term=2, entries=["cmd1"])
    assert success is True
    assert node.state == RaftNode.FOLLOWER
