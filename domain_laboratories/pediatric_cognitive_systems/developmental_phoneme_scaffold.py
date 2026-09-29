"""
Developmental Phoneme Scaffold & Child Language Acquisition Engine
Models child articulatory phonetics, phonological error patterns, Dirichlet-Bayesian
speech interpretation, and Vygotskian Zone of Proximal Development (ZPD) scaffolding.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class PhonemeStage(str, Enum):
    EARLY_8 = "EARLY_8"     # m, b, j, n, w, d, p, h (Typically mastered ages 1.5 - 3)
    MIDDLE_8 = "MIDDLE_8"   # t, ng, k, g, f, v, ch, jh (Typically mastered ages 3 - 5.5)
    LATE_8 = "LATE_8"       # sh, th_unvoiced, s, z, th_voiced, l, r, zh (Ages 5 - 7.5)


class PhonologicalProcess(str, Enum):
    FRONTING = "FRONTING"                     # e.g. /k/ -> [t], /g/ -> [d] (velar to alveolar)
    STOPPING = "STOPPING"                     # e.g. /s/ -> [t], /f/ -> [p] (fricative to stop)
    GLIDING = "GLIDING"                       # e.g. /r/ -> [w], /l/ -> [w] (liquid to glide)
    CLUSTER_REDUCTION = "CLUSTER_REDUCTION"   # e.g. /sp/ -> [p], /tr/ -> [t]
    FINAL_CONSONANT_DELETION = "FINAL_DELETION" # e.g. /bæk/ -> [bæ]
    VOICING_ASSIMILATION = "VOICING_ASSIMILATION"


@dataclass
class ChildSpeechSample:
    intended_word: str
    target_phonemes: List[str]
    produced_phonemes: List[str]
    acoustic_clarity: float  # 0.0 to 1.0
    latency_ms: float


@dataclass
class ScaffoldingRecommendation:
    target_phoneme: str
    cue_type: str  # "visual_articulatory", "tactile_kinesthetic", "auditory_minimal_pair", "aac_symbol"
    scaffold_prompt: str
    minimal_pair_example: Tuple[str, str]
    difficulty_level: float  # 0.0 to 1.0 within child's ZPD
    aac_symbol_glyph: Optional[str] = None


class DevelopmentalPhonemeScaffold:
    """
    Cognitive phonetics engine modeling infant and child speech interpretation,
    tracking developmental trajectories and producing adaptive clinical scaffolding.
    """

    EARLY_PHONEMES = {"m", "b", "j", "n", "w", "d", "p", "h"}
    MIDDLE_PHONEMES = {"t", "ng", "k", "g", "f", "v", "ch", "jh"}
    LATE_PHONEMES = {"sh", "th", "s", "z", "dh", "l", "r", "zh"}

    # Common developmental substitution map: target -> list of typical substitutions
    DEVELOPMENTAL_SUBSTITUTIONS = {
        "k": ["t", "p"],
        "g": ["d", "b"],
        "s": ["t", "th"],
        "z": ["d", "dh"],
        "f": ["p", "b"],
        "v": ["b", "d"],
        "r": ["w", "j"],
        "l": ["w", "j"],
        "sh": ["s", "t"],
        "ch": ["t", "ts"],
        "th": ["f", "t", "s"],
        "dh": ["d", "v"],
    }

    # AAC Symbol Glyphs for Non-Verbal Scaffolding
    AAC_VOCABULARY_GLYPHS = {
        "more": "➕ [MORE]",
        "help": "🤝 [HELP]",
        "stop": "🛑 [STOP]",
        "water": "💧 [DRINK]",
        "pain": "⚡ [PAIN/HURT]",
        "eat": "🍎 [HUNGRY]",
        "break": "⏸️ [SENSORY BREAK]",
        "happy": "😊 [HAPPY]",
        "scared": "😨 [FEAR/OVERWHELMED]",
        "want": "👉 [WANT]",
    }

    def __init__(self, child_age_months: float, is_neurodivergent: bool = False):
        self.age_months = child_age_months
        self.is_neurodivergent = is_neurodivergent

        # Mastery tracking: phoneme -> count of [correct, total]
        self.phoneme_stats: Dict[str, List[int]] = {}
        # Dirichlet conjugate prior counts for Bayesian inference
        self.alpha_prior = 1.0

    def record_utterance(self, sample: ChildSpeechSample) -> Dict[str, float]:
        """
        Record child speech production and update articulatory mastery estimates.
        """
        for t_p, p_p in zip(sample.target_phonemes, sample.produced_phonemes):
            if t_p not in self.phoneme_stats:
                self.phoneme_stats[t_p] = [0, 0]
            self.phoneme_stats[t_p][1] += 1
            if t_p == p_p:
                self.phoneme_stats[t_p][0] += 1

        return self.get_phoneme_accuracies()

    def get_phoneme_accuracies(self) -> Dict[str, float]:
        """
        Bayesian MAP accuracy estimate with Beta(alpha, beta) conjugate prior.
        """
        accuracies = {}
        for p, (correct, total) in self.phoneme_stats.items():
            # Beta(1 + correct, 1 + total - correct)
            acc = (correct + 1.0) / (total + 2.0)
            accuracies[p] = round(float(acc), 3)
        return accuracies

    def detect_phonological_processes(self, target: List[str], produced: List[str]) -> List[PhonologicalProcess]:
        """
        Identify active systematic phonological simplification processes.
        """
        processes = []

        # Final Consonant Deletion
        if len(target) > len(produced) and len(produced) > 0:
            if target[-1] not in {"a", "e", "i", "o", "u"} and produced[-1] in {"a", "e", "i", "o", "u"}:
                processes.append(PhonologicalProcess.FINAL_CONSONANT_DELETION)

        for t_p, p_p in zip(target, produced):
            if t_p == p_p:
                continue

            # Fronting: velar (k, g) replaced by alveolar (t, d)
            if t_p in {"k", "g"} and p_p in {"t", "d"}:
                processes.append(PhonologicalProcess.FRONTING)

            # Stopping: fricative (s, z, f, v, sh, th) replaced by stop (t, d, p, b)
            if t_p in {"s", "z", "f", "v", "sh", "th"} and p_p in {"t", "d", "p", "b"}:
                processes.append(PhonologicalProcess.STOPPING)

            # Gliding: liquid (r, l) replaced by glide (w, j)
            if t_p in {"r", "l"} and p_p in {"w", "j"}:
                processes.append(PhonologicalProcess.GLIDING)

        return list(set(processes))

    def infer_intended_word_bayes(
        self, observed_phonemes: List[str], candidate_lexicon: List[Dict[str, any]]
    ) -> List[Tuple[str, float]]:
        r"""
        Given corrupted/emergent child phonemes, compute posterior probability over
        lexicon candidates P(Word | Observed) \propto P(Observed | Word) * P(Word).
        """
        posteriors = []

        for item in candidate_lexicon:
            word = item["word"]
            target_phonemes = item["phonemes"]
            prior_freq = item.get("prior", 1.0)

            # Likelihood P(Observed | Target)
            log_lik = 0.0
            max_len = max(len(target_phonemes), len(observed_phonemes))

            for idx in range(max_len):
                t_p = target_phonemes[idx] if idx < len(target_phonemes) else "<eps>"
                o_p = observed_phonemes[idx] if idx < len(observed_phonemes) else "<eps>"

                if t_p == o_p:
                    p_match = 0.85
                elif t_p in self.DEVELOPMENTAL_SUBSTITUTIONS and o_p in self.DEVELOPMENTAL_SUBSTITUTIONS[t_p]:
                    # High likelihood for developmentally normative error
                    p_match = 0.55
                elif o_p == "<eps>":  # Deletion
                    p_match = 0.15
                else:  # Random substitution
                    p_match = 0.03

                log_lik += math.log(max(p_match, 1e-6))

            log_posterior = log_lik + math.log(prior_freq)
            posteriors.append((word, log_posterior))

        # Softmax normalization
        max_log = max(p[1] for p in posteriors) if posteriors else 0.0
        exp_scores = [(w, math.exp(score - max_log)) for w, score in posteriors]
        sum_exp = sum(s for _, s in exp_scores)

        normalized = [(w, round(score / max(sum_exp, 1e-12), 4)) for w, score in exp_scores]
        normalized.sort(key=lambda x: x[1], reverse=True)
        return normalized

    def determine_zpd_targets(self) -> List[str]:
        """
        Identify phonemes currently in the child's Zone of Proximal Development (ZPD).
        ZPD is defined as phonemes with accuracy between 20% and 75%, or the next
        developmental stage according to age norms.
        """
        accuracies = self.get_phoneme_accuracies()
        zpd = []

        # Find emerging phonemes (already attempted but not yet mastered)
        for p, acc in accuracies.items():
            if 0.15 <= acc < 0.80:
                zpd.append(p)

        # If few emerging, look at age-normative stages
        if len(zpd) < 3:
            if self.age_months < 36.0:
                candidate_pool = self.EARLY_PHONEMES
            elif self.age_months < 60.0:
                candidate_pool = self.MIDDLE_PHONEMES
            else:
                candidate_pool = self.LATE_PHONEMES

            for p in candidate_pool:
                if p not in accuracies or accuracies[p] < 0.20:
                    if p not in zpd:
                        zpd.append(p)
                    if len(zpd) >= 4:
                        break

        return zpd

    def generate_scaffolding(self, target_phoneme: str) -> ScaffoldingRecommendation:
        """
        Generate multimodal pedagogical scaffold tailored to child's developmental profile.
        """
        minimal_pairs = {
            "k": ("tea", "key"),
            "g": ("dough", "go"),
            "s": ("tea", "see"),
            "f": ("pan", "fan"),
            "r": ("wing", "ring"),
            "l": ("weed", "lead"),
            "sh": ("sip", "ship"),
            "ch": ("tip", "chip"),
            "th": ("thumb", "sum"),
        }

        pair = minimal_pairs.get(target_phoneme, ("pat", f"p{target_phoneme}at"))

        if self.is_neurodivergent:
            # Multi-sensory visual and kinesthetic cueing
            cue = "visual_articulatory"
            prompt = f"Look at my mouth: place tongue back behind teeth. Let's make the quiet sound: /{target_phoneme}/."
        else:
            cue = "auditory_minimal_pair"
            prompt = f"Listen closely: is it '{pair[0]}' or '{pair[1]}'? Notice how /{target_phoneme}/ sounds different!"

        # AAC Glyph lookup if communicative support is active
        aac_glyph = None
        for word, glyph in self.AAC_VOCABULARY_GLYPHS.items():
            if word.startswith(target_phoneme):
                aac_glyph = glyph
                break

        return ScaffoldingRecommendation(
            target_phoneme=target_phoneme,
            cue_type=cue,
            scaffold_prompt=prompt,
            minimal_pair_example=pair,
            difficulty_level=0.55,
            aac_symbol_glyph=aac_glyph,
        )
