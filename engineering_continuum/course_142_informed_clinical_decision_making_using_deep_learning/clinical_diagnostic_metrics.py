"""Course 142: Clinical Diagnostic Sensitivity, Specificity & Youden's J Index"""
class ClinicalDiagnosticMetrics:
    @staticmethod
    def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden_j = sensitivity + specificity - 1.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "youden_j": youden_j,
            "f1_score": f1
        }
