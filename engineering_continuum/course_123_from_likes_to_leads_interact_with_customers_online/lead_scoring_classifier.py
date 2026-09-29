"""Course 123: Logistic Lead Scoring & Qualification Tiering Classifier"""
import math

class LeadScoringClassifier:
    def __init__(self, weights: dict, bias: float = -2.0):
        self.weights = weights
        self.bias = bias

    def predict_probability(self, features: dict) -> float:
        z = self.bias + sum(self.weights.get(k, 0.0) * features.get(k, 0.0) for k in self.weights)
        return float(1.0 / (1.0 + math.exp(-z)))

    def assign_tier(self, features: dict) -> str:
        prob = self.predict_probability(features)
        if prob >= 0.8:
            return "MQL_HOT"
        elif prob >= 0.5:
            return "MQL_WARM"
        return "NURTURE"
