from viterbi_pos_tagger import ViterbiPOSTagger

def test_viterbi():
    states = ["NOUN", "VERB"]
    start = {"NOUN": 0.8, "VERB": 0.2}
    trans = {"NOUN": {"NOUN": 0.1, "VERB": 0.9}, "VERB": {"NOUN": 0.8, "VERB": 0.2}}
    emit = {"NOUN": {"runs": 0.1, "dog": 0.8}, "VERB": {"runs": 0.8, "dog": 0.1}}
    
    path = ViterbiPOSTagger.decode(["dog", "runs"], states, start, trans, emit)
    assert path == ["NOUN", "VERB"]
