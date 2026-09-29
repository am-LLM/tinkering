"""Course 164: RFM (Recency, Frequency, Monetary) Customer Segmentation Engine"""
import numpy as np

class RFMCustomerSegmentation:
    @staticmethod
    def score_rfm(recency_days: list, frequency_counts: list, monetary_vals: list) -> list:
        n = len(recency_days)
        # 1-5 scoring
        r_ranks = np.argsort(np.argsort(-np.array(recency_days))) + 1
        f_ranks = np.argsort(np.argsort(np.array(frequency_counts))) + 1
        m_ranks = np.argsort(np.argsort(np.array(monetary_vals))) + 1
        
        scores = []
        for i in range(n):
            scores.append({
                "r_score": int(np.ceil(r_ranks[i] * 5.0 / n)),
                "f_score": int(np.ceil(f_ranks[i] * 5.0 / n)),
                "m_score": int(np.ceil(m_ranks[i] * 5.0 / n))
            })
        return scores
