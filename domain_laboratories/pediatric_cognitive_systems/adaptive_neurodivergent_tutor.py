"""
Adaptive Neurodivergent Tutor & Special Education Learning Engine
Implements Structured TEACCH, Discrete Trial Training (DTT), POMDP cognitive state tracking,
dynamic sensory load management, and reinforcement schedules for neurodivergent pediatric learners.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class LearnerState(str, Enum):
    ENGAGED_OPTIMAL = "ENGAGED_OPTIMAL"
    SENSORY_FATIGUE = "SENSORY_FATIGUE"
    COGNITIVELY_OVERLOADED = "COGNITIVELY_OVERLOADED"
    DISSOCIATED_SHUTDOWN = "DISSOCIATED_SHUTDOWN"


class PromptLevel(int, Enum):
    INDEPENDENT = 0       # No prompt needed
    VERBAL_INDIRECT = 1   # "What's next?"
    VERBAL_DIRECT = 2     # "Touch the blue square"
    GESTURAL = 3          # Pointing / nodding toward stimulus
    MODELING = 4          # Demonstrating the action
    PARTIAL_PHYSICAL = 5  # Gentle elbow guidance
    FULL_PHYSICAL = 6     # Hand-over-hand assistance


@dataclass
class TrialResult:
    trial_id: int
    task_difficulty: float      # 0.0 to 1.0
    prompt_used: PromptLevel
    response_correct: bool
    response_latency_sec: float
    perseveration_detected: bool = False
    motor_fidget_score: float = 0.1  # 0.0 to 1.0


@dataclass
class TutorAction:
    action_type: str  # "PRESENT_TASK", "SENSORY_BREAK", "ADJUST_PROMPT", "AWARD_REINFORCER"
    prompt_level: PromptLevel
    target_difficulty: float
    sensory_break_duration_s: float
    tokens_awarded: int
    rationale: str


class AdaptiveNeurodivergentTutor:
    """
    Pedagogical executive optimizing cognitive scaffolding, sensory homeostasis,
    and Discrete Trial Training (DTT) for pediatric neurodivergent learners.
    """

    def __init__(
        self,
        sensory_capacity_budget: float = 100.0,
        reinforcement_schedule: str = "VR_3",  # Fixed Ratio (FR_1, FR_3) or Variable Ratio (VR_3)
        target_token_goal: int = 5,
    ):
        self.max_sensory_capacity = sensory_capacity_budget
        self.current_sensory_load = 0.0  # Accumulates with trials, decays with breaks
        self.reinforcement_schedule = reinforcement_schedule
        self.tokens_collected = 0
        self.token_goal = target_token_goal

        # POMDP belief state over LearnerState
        # [ENGAGED_OPTIMAL, SENSORY_FATIGUE, COGNITIVELY_OVERLOADED, DISSOCIATED_SHUTDOWN]
        self.belief = np.array([0.70, 0.15, 0.10, 0.05], dtype=np.float64)

        # Performance history
        self.trials: List[TrialResult] = []
        self.consecutive_errors = 0
        self.consecutive_successes = 0
        self.vr_counter = 0
        self.vr_threshold = 3

    def update_pomdp_belief(self, trial: TrialResult) -> np.ndarray:
        """
        Bayesian observation update of learner's latent cognitive state given trial telemetry.
        """
        # Observation likelihood matrix P(Observation | State)
        # Observations: (Correct, Latency, Fidget)
        is_slow = trial.response_latency_sec > 4.5
        is_fast = trial.response_latency_sec < 2.0
        high_fidget = trial.motor_fidget_score > 0.65

        # Likelihood vectors for each state
        # 0: ENGAGED_OPTIMAL
        p_optimal = (0.85 if trial.response_correct else 0.15) * (0.80 if is_fast else 0.20) * (0.15 if high_fidget else 0.85)
        # 1: SENSORY_FATIGUE
        p_fatigue = (0.50 if trial.response_correct else 0.50) * (0.70 if is_slow else 0.30) * (0.60 if high_fidget else 0.40)
        # 2: COGNITIVELY_OVERLOADED
        p_overload = (0.20 if trial.response_correct else 0.80) * (0.85 if is_slow else 0.15) * (0.85 if high_fidget else 0.15)
        # 3: DISSOCIATED_SHUTDOWN
        p_shutdown = (0.05 if trial.response_correct else 0.95) * (0.95 if is_slow else 0.05) * (0.90 if trial.perseveration_detected else 0.20)

        likelihood = np.array([p_optimal, p_fatigue, p_overload, p_shutdown], dtype=np.float64)
        likelihood = np.maximum(likelihood, 1e-6)

        # Markov transition matrix (spontaneous fatigue drift)
        T = np.array([
            [0.85, 0.08, 0.05, 0.02],
            [0.10, 0.75, 0.10, 0.05],
            [0.05, 0.15, 0.70, 0.10],
            [0.02, 0.08, 0.10, 0.80],
        ], dtype=np.float64)

        # Prior projection: belief * T
        prior = np.dot(self.belief, T)

        # Posterior update: P(State | Obs) \propto P(Obs | State) * Prior
        posterior = prior * likelihood
        self.belief = posterior / np.sum(posterior)
        return self.belief

    def get_dominant_state(self) -> Tuple[LearnerState, float]:
        """
        Return the maximum a posteriori (MAP) cognitive state and its probability.
        """
        states = [
            LearnerState.ENGAGED_OPTIMAL,
            LearnerState.SENSORY_FATIGUE,
            LearnerState.COGNITIVELY_OVERLOADED,
            LearnerState.DISSOCIATED_SHUTDOWN,
        ]
        best_idx = int(np.argmax(self.belief))
        return states[best_idx], float(self.belief[best_idx])

    def record_trial(self, trial: TrialResult) -> TutorAction:
        """
        Process trial outcome, update sensory/cognitive state, and generate the next clinical action.
        """
        self.trials.append(trial)
        self.update_pomdp_belief(trial)

        # Update sensory load: cognitive trial adds load, high latency/fidget adds extra
        added_load = 5.0 + (trial.task_difficulty * 5.0) + (10.0 if trial.motor_fidget_score > 0.6 else 0.0)
        self.current_sensory_load = min(self.max_sensory_capacity, self.current_sensory_load + added_load)

        if trial.response_correct:
            self.consecutive_successes += 1
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
            self.consecutive_successes = 0

        dom_state, state_prob = self.get_dominant_state()

        # Decision Logic:
        # 1. Extreme sensory overload or shutdown -> Immediate Sensory Break
        if dom_state == LearnerState.DISSOCIATED_SHUTDOWN or self.current_sensory_load > (0.85 * self.max_sensory_capacity):
            break_time = 45.0 if dom_state == LearnerState.DISSOCIATED_SHUTDOWN else 20.0
            self.current_sensory_load = max(0.0, self.current_sensory_load - 40.0)
            self.belief = np.array([0.50, 0.30, 0.15, 0.05])  # Reset partially
            return TutorAction(
                action_type="SENSORY_BREAK",
                prompt_level=PromptLevel.INDEPENDENT,
                target_difficulty=max(0.1, trial.task_difficulty - 0.25),
                sensory_break_duration_s=break_time,
                tokens_awarded=0,
                rationale="Autonomic/cognitive sensory load threshold exceeded; dispensing proprioceptive/vestibular break.",
            )

        # 2. Cognitive Overload -> Prompt fading back up (Most-to-Least support) and lower difficulty
        if dom_state == LearnerState.COGNITIVELY_OVERLOADED or self.consecutive_errors >= 2:
            new_prompt = PromptLevel(min(int(PromptLevel.FULL_PHYSICAL), int(trial.prompt_used) + 2))
            new_diff = max(0.1, trial.task_difficulty - 0.15)
            return TutorAction(
                action_type="ADJUST_PROMPT",
                prompt_level=new_prompt,
                target_difficulty=round(new_diff, 2),
                sensory_break_duration_s=0.0,
                tokens_awarded=0,
                rationale="Error streak detected; escalating prompt hierarchy and scaling task complexity down.",
            )

        # 3. Successful Trial -> Check Reinforcement Schedule & Token Economy
        tokens_to_award = 0
        if trial.response_correct:
            if self.reinforcement_schedule == "FR_1":
                tokens_to_award = 1
            elif self.reinforcement_schedule == "VR_3":
                self.vr_counter += 1
                if self.vr_counter >= self.vr_threshold:
                    tokens_to_award = 1
                    self.vr_counter = 0
                    self.vr_threshold = np.random.choice([2, 3, 4])

            self.tokens_collected += tokens_to_award

            # Next trial parameters: fade prompt toward independence if successful
            new_prompt = PromptLevel(max(int(PromptLevel.INDEPENDENT), int(trial.prompt_used) - 1))
            new_diff = min(1.0, trial.task_difficulty + (0.1 if self.consecutive_successes >= 3 else 0.0))

            return TutorAction(
                action_type="AWARD_REINFORCER" if tokens_to_award > 0 else "PRESENT_TASK",
                prompt_level=new_prompt,
                target_difficulty=round(new_diff, 2),
                sensory_break_duration_s=0.0,
                tokens_awarded=tokens_to_award,
                rationale=f"Correct response registered; tokens={self.tokens_collected}/{self.token_goal}, prompt faded.",
            )

        # Default standard continuation
        return TutorAction(
            action_type="PRESENT_TASK",
            prompt_level=trial.prompt_used,
            target_difficulty=trial.task_difficulty,
            sensory_break_duration_s=0.0,
            tokens_awarded=0,
            rationale="Continuing active discrete trial pacing.",
        )
