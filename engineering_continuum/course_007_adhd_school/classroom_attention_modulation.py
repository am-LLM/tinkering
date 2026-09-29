import numpy as np

class AttentionModulationEngine:
    def __init__(self, baseline_tonic_dopamine: float = 0.5):
        self.tonic_da = baseline_tonic_dopamine

    def step_attention(self, stimulus_salience: float, task_monotony_sec: float) -> float:
        phasic_burst = np.tanh(stimulus_salience * 2.0)
        monotony_decay = np.exp(-task_monotony_sec / 300.0)
        attention_level = (self.tonic_da * 0.4 + phasic_burst * 0.6) * monotony_decay
        return float(np.clip(attention_level, 0.0, 1.0))
