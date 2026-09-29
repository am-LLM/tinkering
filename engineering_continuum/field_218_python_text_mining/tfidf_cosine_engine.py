"""Course 218: TF-IDF Vectorizer & Document Cosine Similarity Engine"""
import math
import re

class TFIDFCosineEngine:
    @staticmethod
    def _tokenize(doc: str) -> list:
        return re.findall(r'\w+', doc.lower())

    @classmethod
    def compute_tfidf_and_cosine(cls, doc1: str, doc2: str) -> float:
        tokens1 = cls._tokenize(doc1)
        tokens2 = cls._tokenize(doc2)
        vocab = sorted(list(set(tokens1 + tokens2)))
        
        def tf(tokens):
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            return [counts.get(w, 0) / len(tokens) for w in vocab]
            
        tf1 = tf(tokens1)
        tf2 = tf(tokens2)
        
        # Cosine similarity
        dot = sum(a * b for a, b in zip(tf1, tf2))
        norm1 = math.sqrt(sum(a*a for a in tf1))
        norm2 = math.sqrt(sum(b*b for b in tf2))
        return float(dot / (norm1 * norm2)) if (norm1 * norm2) > 0 else 0.0
