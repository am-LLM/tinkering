"""Course 224: Rate Monotonic (RMS) & Earliest Deadline First (EDF) Schedulability Analyzer"""
class RealTimeSchedulability:
    @staticmethod
    def rms_utilization_bound(n_tasks: int) -> float:
        # Liu & Layland bound: n * (2^(1/n) - 1)
        if n_tasks <= 0:
            return 0.0
        return float(n_tasks * (2.0 ** (1.0 / n_tasks) - 1.0))

    @classmethod
    def check_rms_schedulable(cls, execution_times: list, periods: list) -> bool:
        total_u = sum(c / p for c, p in zip(execution_times, periods))
        bound = cls.rms_utilization_bound(len(execution_times))
        return total_u <= bound

    @staticmethod
    def check_edf_schedulable(execution_times: list, periods: list) -> bool:
        # EDF is optimal on uniprocessor if U <= 1.0
        total_u = sum(c / p for c, p in zip(execution_times, periods))
        return total_u <= 1.0
