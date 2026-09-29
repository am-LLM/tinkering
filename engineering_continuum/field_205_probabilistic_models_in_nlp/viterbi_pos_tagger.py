"""Course 205: Hidden Markov Model (HMM) Viterbi Part-of-Speech Tag Sequence Decoder"""
import numpy as np

class ViterbiPOSTagger:
    @staticmethod
    def decode(observations: list, states: list, start_p: dict, trans_p: dict, emit_p: dict) -> list:
        n_obs = len(observations)
        n_states = len(states)
        v = [{s: {"prob": start_p.get(s, 1e-6) * emit_p[s].get(observations[0], 1e-6), "prev": None} for s in states}]
        
        for t in range(1, n_obs):
            v.append({})
            for curr_s in states:
                best_prob = -1.0
                best_prev = None
                for prev_s in states:
                    prob = v[t-1][prev_s]["prob"] * trans_p[prev_s].get(curr_s, 1e-6) * emit_p[curr_s].get(observations[t], 1e-6)
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = prev_s
                v[t][curr_s] = {"prob": best_prob, "prev": best_prev}
                
        # Backtrack
        best_last_state = max(states, key=lambda s: v[-1][s]["prob"])
        path = [best_last_state]
        for t in range(n_obs - 1, 0, -1):
            path.append(v[t][path[-1]]["prev"])
        path.reverse()
        return path
