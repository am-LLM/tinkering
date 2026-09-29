"""
Pediatric Cognitive Systems & Developmental AI Cluster
Includes:
- Infant Cry Acoustic Analyzer (DSP, F0 HPS, LPC formants, distress classification)
- Developmental Phoneme Scaffold (ZPD acquisition, Bayesian confusion matrix, phonological processes)
- Adaptive Neurodivergent Tutor (TEACCH/DTT POMDP, sensory load balancing, token economy)
- Extreme Pediatric Crisis Engine (ANS meltdown early warning, locked-in P300 BCI, trauma de-escalation)
"""

from .infant_cry_acoustic_analyzer import InfantCryAnalyzer, CryDistressClassification, AcousticFeatures
from .developmental_phoneme_scaffold import DevelopmentalPhonemeScaffold, PhonemeStage, ChildSpeechSample
from .adaptive_neurodivergent_tutor import AdaptiveNeurodivergentTutor, LearnerState, TrialResult, PromptLevel
from .extreme_pediatric_crisis_engine import ExtremePediatricCrisisEngine, AutonomicState, CrisisSeverity, BCIResponse

__all__ = [
    "InfantCryAnalyzer",
    "CryDistressClassification",
    "AcousticFeatures",
    "DevelopmentalPhonemeScaffold",
    "PhonemeStage",
    "ChildSpeechSample",
    "AdaptiveNeurodivergentTutor",
    "LearnerState",
    "TrialResult",
    "PromptLevel",
    "ExtremePediatricCrisisEngine",
    "AutonomicState",
    "CrisisSeverity",
    "BCIResponse",
]
