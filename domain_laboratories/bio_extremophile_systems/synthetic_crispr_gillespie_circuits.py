"""
Synthetic CRISPR-dCas9 Gillespie Stochastic Biochemical Circuits Simulator.

Implements the Stochastic Simulation Algorithm (SSA / Gillespie Direct Method)
for synthetic genetic networks including:
1. Mutual-repression Genetic Toggle Switch with CRISPR-dCas9 transcriptional silencing.
2. 3-node Cyclical Repressilator with dCas9/guide-RNA mediated repression.
3. Generalized Chemical Master Equation (CME) stochastic circuit engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class CircuitType(Enum):
    TOGGLE_SWITCH = "toggle_switch"
    REPRESSILATOR = "repressilator"
    CUSTOM = "custom"


@dataclass
class Reaction:
    """Represents a single biochemical reaction channel."""
    name: str
    reactants: Dict[str, int]  # species name -> stoichiometric coefficient consumed
    products: Dict[str, int]   # species name -> stoichiometric coefficient produced
    propensity_fn: Callable[[Dict[str, int]], float]


@dataclass
class ToggleSwitchParams:
    """Kinetic parameters for CRISPR-dCas9 genetic toggle switch."""
    # Transcription rates (molecules/min)
    k_tx_u: float = 12.0
    k_tx_v: float = 12.0
    # Translation rates (proteins/mRNA/min)
    k_tl_u: float = 8.0
    k_tl_v: float = 8.0
    # mRNA degradation rate constants (1/min)
    d_mrna_u: float = 0.25
    d_mrna_v: float = 0.25
    # Protein degradation rate constants (1/min)
    d_prot_u: float = 0.05
    d_prot_v: float = 0.05
    # dCas9:gRNA repression parameters
    # Hill coefficient for cooperativity / dCas9 effective repression binding
    n_hill: float = 2.5
    # Threshold protein copy number for 50% repression (theta / K_d)
    k_rep: float = 30.0
    # Basal leakiness fraction (fraction of transcription active under full repression)
    leak_fraction: float = 0.01


@dataclass
class RepressilatorParams:
    """Kinetic parameters for 3-node CRISPR-dCas9 repressilator."""
    # Node X -> Node Y -> Node Z -> Node X
    k_tx: float = 15.0         # Transcription rate (molecules/min)
    k_tl: float = 10.0         # Translation rate (proteins/mRNA/min)
    d_mrna: float = 0.3        # mRNA degradation rate (1/min)
    d_prot: float = 0.04       # Protein degradation rate (1/min)
    n_hill: float = 3.0        # CRISPR/dCas9 effective cooperativity
    k_rep: float = 25.0        # Repression threshold copy number
    leak_fraction: float = 0.005


@dataclass
class GillespieSimulationResult:
    """Container for stochastic trajectory results."""
    time: np.ndarray
    state_trajectories: Dict[str, np.ndarray]
    reaction_counts: Dict[str, int]
    circuit_type: CircuitType
    final_time: float
    total_steps: int

    def get_species_final(self, species: str) -> int:
        if species in self.state_trajectories:
            return int(self.state_trajectories[species][-1])
        raise KeyError(f"Species '{species}' not found in simulation results.")

    def calculate_toggle_bistability_ratio(self) -> float:
        """Returns log10 ratio of Protein U to Protein V at final state."""
        pu = max(1, self.get_species_final("P_u"))
        pv = max(1, self.get_species_final("P_v"))
        return float(np.log10(pu / pv))

    def detect_repressilator_oscillations(
        self, species: str = "P_x", min_peaks: int = 2
    ) -> Tuple[bool, float, float]:
        """
        Detects sustained oscillations in repressilator trajectory.
        Returns: (is_oscillating, estimated_period, mean_amplitude)
        """
        if species not in self.state_trajectories:
            raise KeyError(f"Species '{species}' not found in results.")

        t = self.time
        signal = self.state_trajectories[species]

        # Ignore initial transient (first 25%)
        transient_idx = int(len(t) * 0.25)
        if len(t) - transient_idx < 20:
            return False, 0.0, 0.0

        t_sub = t[transient_idx:]
        sig_sub = signal[transient_idx:]

        # Peak detection via local maxima
        peak_indices = []
        for i in range(1, len(sig_sub) - 1):
            if sig_sub[i] > sig_sub[i - 1] and sig_sub[i] > sig_sub[i + 1]:
                if sig_sub[i] > np.mean(sig_sub):
                    peak_indices.append(i)

        if len(peak_indices) < min_peaks:
            return False, 0.0, 0.0

        peak_times = t_sub[peak_indices]
        periods = np.diff(peak_times)
        avg_period = float(np.mean(periods)) if len(periods) > 0 else 0.0

        # Calculate amplitude
        trough_values = []
        for i in range(1, len(sig_sub) - 1):
            if sig_sub[i] < sig_sub[i - 1] and sig_sub[i] < sig_sub[i + 1]:
                trough_values.append(sig_sub[i])

        peak_values = sig_sub[peak_indices]
        mean_peak = np.mean(peak_values)
        mean_trough = np.mean(trough_values) if trough_values else np.min(sig_sub)
        amplitude = float((mean_peak - mean_trough) / 2.0)

        is_oscillating = len(peak_indices) >= min_peaks and avg_period > 0 and amplitude > 5.0
        return is_oscillating, avg_period, amplitude


class GillespieCircuitEngine:
    """
    Exact Gillespie Stochastic Simulation Algorithm (SSA Direct Method)
    for biochemical networks governed by the Chemical Master Equation.
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = np.random.default_rng(rng_seed)
        self.reactions: List[Reaction] = []
        self.species_initial: Dict[str, int] = {}

    def set_initial_state(self, initial_state: Dict[str, int]) -> None:
        """Sets the initial molecule counts for all species."""
        for k, v in initial_state.items():
            if v < 0:
                raise ValueError(f"Species count cannot be negative: {k}={v}")
        self.species_initial = dict(initial_state)

    def add_reaction(self, reaction: Reaction) -> None:
        """Registers a reaction channel into the network."""
        self.reactions.append(reaction)

    def clear(self) -> None:
        """Clears all registered reactions and species."""
        self.reactions.clear()
        self.species_initial.clear()

    def run_simulation(
        self,
        t_max: float,
        max_steps: int = 500000,
        record_interval: Optional[float] = None,
        circuit_type: CircuitType = CircuitType.CUSTOM,
    ) -> GillespieSimulationResult:
        """
        Executes Gillespie Direct Method trajectory until t >= t_max or max_steps reached.
        """
        if t_max <= 0:
            raise ValueError(f"t_max must be positive, got {t_max}")
        if not self.reactions:
            raise RuntimeError("No reactions defined in Gillespie engine.")

        # Initialize current state
        state = dict(self.species_initial)
        for rxn in self.reactions:
            for s in list(rxn.reactants.keys()) + list(rxn.products.keys()):
                if s not in state:
                    state[s] = 0

        species_list = sorted(list(state.keys()))
        reaction_counts = {rxn.name: 0 for rxn in self.reactions}

        # Trajectory recording storage
        time_points: List[float] = [0.0]
        trajectories: Dict[str, List[int]] = {s: [state[s]] for s in species_list}

        current_time = 0.0
        step = 0
        next_record_time = record_interval if record_interval is not None else None

        while current_time < t_max and step < max_steps:
            # 1. Compute propensities a_j for all reactions
            propensities = np.array(
                [max(0.0, rxn.propensity_fn(state)) for rxn in self.reactions],
                dtype=np.float64,
            )
            a_0 = float(np.sum(propensities))

            if a_0 <= 1e-14:
                # Absorbing state or zero propensity
                current_time = t_max
                if next_record_time is not None and next_record_time <= t_max:
                    time_points.append(current_time)
                    for s in species_list:
                        trajectories[s].append(state[s])
                break

            # 2. Draw random numbers r1, r2 ~ Uniform(0, 1)
            r1 = self.rng.random()
            r2 = self.rng.random()

            # 3. Calculate time to next reaction tau ~ Exp(a_0)
            tau = -np.log(max(1e-15, r1)) / a_0
            next_event_time = current_time + tau

            # If recording at discrete intervals, interpolate/hold state
            if next_record_time is not None:
                while next_record_time is not None and next_record_time <= min(next_event_time, t_max):
                    time_points.append(next_record_time)
                    for s in species_list:
                        trajectories[s].append(state[s])
                    next_record_time += record_interval

            # 4. Select reaction channel mu using cumulative sum
            cum_prop = np.cumsum(propensities)
            threshold = r2 * a_0
            mu_idx = int(np.searchsorted(cum_prop, threshold))
            if mu_idx >= len(self.reactions):
                mu_idx = len(self.reactions) - 1

            chosen_rxn = self.reactions[mu_idx]

            # 5. Update state vector
            for reactant, stoichiometry in chosen_rxn.reactants.items():
                state[reactant] = max(0, state[reactant] - stoichiometry)

            for product, stoichiometry in chosen_rxn.products.items():
                state[product] = state[product] + stoichiometry

            reaction_counts[chosen_rxn.name] += 1
            current_time = next_event_time
            step += 1

            # If not using fixed record intervals, save every step
            if record_interval is None:
                time_points.append(current_time)
                for s in species_list:
                    trajectories[s].append(state[s])

        # Ensure final state recorded
        if record_interval is not None and (len(time_points) == 0 or time_points[-1] < t_max):
            time_points.append(t_max)
            for s in species_list:
                trajectories[s].append(state[s])

        time_arr = np.array(time_points, dtype=np.float64)
        traj_arr = {s: np.array(trajectories[s], dtype=np.int32) for s in species_list}

        return GillespieSimulationResult(
            time=time_arr,
            state_trajectories=traj_arr,
            reaction_counts=reaction_counts,
            circuit_type=circuit_type,
            final_time=current_time,
            total_steps=step,
        )


def build_crispr_toggle_switch(
    params: ToggleSwitchParams,
    initial_counts: Optional[Dict[str, int]] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[GillespieCircuitEngine, Dict[str, int]]:
    """
    Constructs a CRISPR-dCas9 Genetic Toggle Switch model.
    Gene U produces mRNA_u -> Protein U (guides dCas9 to repress Promoter V).
    Gene V produces mRNA_v -> Protein V (guides dCas9 to repress Promoter U).
    """
    engine = GillespieCircuitEngine(rng_seed=rng_seed)

    if initial_counts is None:
        initial_counts = {
            "M_u": 10,
            "P_u": 100,
            "M_v": 0,
            "P_v": 0,
        }

    engine.set_initial_state(initial_counts)

    # 1. Transcription of Gene U (repressed by Protein V)
    def prop_tx_u(st: Dict[str, int]) -> float:
        pv = st.get("P_v", 0)
        repression = (pv / params.k_rep) ** params.n_hill
        act_factor = params.leak_fraction + (1.0 - params.leak_fraction) / (1.0 + repression)
        return float(params.k_tx_u * act_factor)

    engine.add_reaction(
        Reaction(
            name="Tx_U",
            reactants={},
            products={"M_u": 1},
            propensity_fn=prop_tx_u,
        )
    )

    # 2. Transcription of Gene V (repressed by Protein U)
    def prop_tx_v(st: Dict[str, int]) -> float:
        pu = st.get("P_u", 0)
        repression = (pu / params.k_rep) ** params.n_hill
        act_factor = params.leak_fraction + (1.0 - params.leak_fraction) / (1.0 + repression)
        return float(params.k_tx_v * act_factor)

    engine.add_reaction(
        Reaction(
            name="Tx_V",
            reactants={},
            products={"M_v": 1},
            propensity_fn=prop_tx_v,
        )
    )

    # 3. Translation of mRNA U -> Protein U
    engine.add_reaction(
        Reaction(
            name="Tl_U",
            reactants={},
            products={"P_u": 1},
            propensity_fn=lambda st: float(params.k_tl_u * st.get("M_u", 0)),
        )
    )

    # 4. Translation of mRNA V -> Protein V
    engine.add_reaction(
        Reaction(
            name="Tl_V",
            reactants={},
            products={"P_v": 1},
            propensity_fn=lambda st: float(params.k_tl_v * st.get("M_v", 0)),
        )
    )

    # 5. Degradation of mRNA U
    engine.add_reaction(
        Reaction(
            name="Deg_M_u",
            reactants={"M_u": 1},
            products={},
            propensity_fn=lambda st: float(params.d_mrna_u * st.get("M_u", 0)),
        )
    )

    # 6. Degradation of mRNA V
    engine.add_reaction(
        Reaction(
            name="Deg_M_v",
            reactants={"M_v": 1},
            products={},
            propensity_fn=lambda st: float(params.d_mrna_v * st.get("M_v", 0)),
        )
    )

    # 7. Degradation of Protein U
    engine.add_reaction(
        Reaction(
            name="Deg_P_u",
            reactants={"P_u": 1},
            products={},
            propensity_fn=lambda st: float(params.d_prot_u * st.get("P_u", 0)),
        )
    )

    # 8. Degradation of Protein V
    engine.add_reaction(
        Reaction(
            name="Deg_P_v",
            reactants={"P_v": 1},
            products={},
            propensity_fn=lambda st: float(params.d_prot_v * st.get("P_v", 0)),
        )
    )

    return engine, initial_counts


def build_crispr_repressilator(
    params: RepressilatorParams,
    initial_counts: Optional[Dict[str, int]] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[GillespieCircuitEngine, Dict[str, int]]:
    """
    Constructs a 3-node CRISPR-dCas9 Cyclical Repressilator.
    Topology: Gene X represses Gene Y, Gene Y represses Gene Z, Gene Z represses Gene X.
    """
    engine = GillespieCircuitEngine(rng_seed=rng_seed)

    if initial_counts is None:
        initial_counts = {
            "M_x": 10,
            "P_x": 100,
            "M_y": 0,
            "P_y": 0,
            "M_z": 0,
            "P_z": 0,
        }

    engine.set_initial_state(initial_counts)

    nodes = [
        ("x", "z"),  # Node X repressed by Node Z
        ("y", "x"),  # Node Y repressed by Node X
        ("z", "y"),  # Node Z repressed by Node Y
    ]

    for target, repressor in nodes:
        # Transcription
        def make_tx_prop(rep_node: str):
            def prop_tx(st: Dict[str, int]) -> float:
                p_rep = st.get(f"P_{rep_node}", 0)
                rep = (p_rep / params.k_rep) ** params.n_hill
                act_factor = params.leak_fraction + (1.0 - params.leak_fraction) / (1.0 + rep)
                return float(params.k_tx * act_factor)
            return prop_tx

        engine.add_reaction(
            Reaction(
                name=f"Tx_{target.upper()}",
                reactants={},
                products={f"M_{target}": 1},
                propensity_fn=make_tx_prop(repressor),
            )
        )

        # Translation
        def make_tl_prop(tgt: str):
            return lambda st: float(params.k_tl * st.get(f"M_{tgt}", 0))

        engine.add_reaction(
            Reaction(
                name=f"Tl_{target.upper()}",
                reactants={},
                products={f"P_{target}": 1},
                propensity_fn=make_tl_prop(target),
            )
        )

        # mRNA degradation
        def make_mrna_deg(tgt: str):
            return lambda st: float(params.d_mrna * st.get(f"M_{tgt}", 0))

        engine.add_reaction(
            Reaction(
                name=f"Deg_M_{target}",
                reactants={f"M_{target}": 1},
                products={},
                propensity_fn=make_mrna_deg(target),
            )
        )

        # Protein degradation
        def make_prot_deg(tgt: str):
            return lambda st: float(params.d_prot * st.get(f"P_{tgt}", 0))

        engine.add_reaction(
            Reaction(
                name=f"Deg_P_{target}",
                reactants={f"P_{target}": 1},
                products={},
                propensity_fn=make_prot_deg(target),
            )
        )

    return engine, initial_counts


def simulate_toggle_switch(
    t_max: float = 200.0,
    record_interval: float = 1.0,
    params: Optional[ToggleSwitchParams] = None,
    initial_counts: Optional[Dict[str, int]] = None,
    rng_seed: Optional[int] = 42,
) -> GillespieSimulationResult:
    """Convenience wrapper to build and simulate a CRISPR toggle switch."""
    p = params or ToggleSwitchParams()
    engine, _ = build_crispr_toggle_switch(p, initial_counts=initial_counts, rng_seed=rng_seed)
    return engine.run_simulation(
        t_max=t_max,
        record_interval=record_interval,
        circuit_type=CircuitType.TOGGLE_SWITCH,
    )


def simulate_repressilator(
    t_max: float = 300.0,
    record_interval: float = 1.0,
    params: Optional[RepressilatorParams] = None,
    initial_counts: Optional[Dict[str, int]] = None,
    rng_seed: Optional[int] = 42,
) -> GillespieSimulationResult:
    """Convenience wrapper to build and simulate a CRISPR repressilator."""
    p = params or RepressilatorParams()
    engine, _ = build_crispr_repressilator(p, initial_counts=initial_counts, rng_seed=rng_seed)
    return engine.run_simulation(
        t_max=t_max,
        record_interval=record_interval,
        circuit_type=CircuitType.REPRESSILATOR,
    )
