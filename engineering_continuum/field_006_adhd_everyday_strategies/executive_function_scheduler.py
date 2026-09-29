import numpy as np

class ExecutiveFunctionScheduler:
    def __init__(self, working_memory_span: int = 4):
        self.span = working_memory_span

    def chunk_tasks(self, tasks: list, max_duration_min: int = 25) -> list:
        chunks = []
        for t in tasks:
            name, dur = t["name"], t["duration"]
            n_chunks = max(1, int(np.ceil(dur / max_duration_min)))
            for i in range(n_chunks):
                chunks.append({"subtask": f"{name}_part_{i+1}", "duration": min(dur, max_duration_min)})
        return chunks

    def compute_cognitive_fatigue(self, work_chunks_completed: int, break_ratio: float = 0.2) -> float:
        decay = np.exp(-0.05 * work_chunks_completed * (1.0 - break_ratio))
        return float(np.clip(1.0 - decay, 0.0, 1.0))
