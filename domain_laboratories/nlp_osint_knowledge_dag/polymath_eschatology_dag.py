"""
Polymath Eschatology Directed Acyclic Graph (DAG) & Geopolitical Event Correlator.

Features:
- Predictive Narrative Ontologies across 4 major traditions:
  - Islamic (Qiyamah, Mahdi, Al-Masih ad-Dajjal, Isa descent, Malahim, Yajuj wa Majuj).
  - Rabbinic / Judaic (Acharit haYamim, Chevlei Mashiach, Mashiach ben Yosef/David, War of Gog u'Magog, Third Temple).
  - Vedic / Sanatana Dharma (Kali Yuga decline, Adharma apex, Kalki Avatar descent, Krita/Satya Yuga restoration).
  - Biblical / Christian (Revelation, Tribulation, Antichrist/Beast, Battle of Armageddon, Parousia, Millennium).
- Formal DAG Graph Theory Engine:
  - Cycle detection, deterministic topological sorting, ancestor/descendant reachability.
- Cross-Corpus Thematic Semantic Affinity & Distance Matrix:
  - Computes archetypal congruence across eschatological Deliverer, Deceiver, Cataclysm, and Renewal paradigms.
- Geopolitical Event Correlation & Narrative Weaponization Anomaly Detector:
  - Temporal alignment of real-world crises (e.g. Euphrates depletion, Levant conflicts, Red Sea chokepoints, AI panopticon)
    with eschatological milestone triggers.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np


class Tradition(Enum):
    ISLAMIC = "Islamic"
    RABBINIC = "Rabbinic"
    VEDIC = "Vedic"
    BIBLICAL = "Biblical"


class EschatologicalArchetype(Enum):
    MORAL_DECAY_PRECURSOR = "MoralDecayPrecursor"
    COSMIC_ENVIRONMENTAL_SIGN = "CosmicEnvironmentalSign"
    GEOPOLITICAL_TRIBULATION = "GeopoliticalTribulation"
    ARCH_DECEIVER_TYRANT = "ArchDeceiverTyrant"
    MESSIANIC_DELIVERER = "MessianicDeliverer"
    COSMIC_CLIMACTIC_BATTLE = "CosmicClimacticBattle"
    GOLDEN_AGE_RESTORATION = "GoldenAgeRestoration"


@dataclass
class EschatologyNode:
    """A discrete prophetic motif or milestone in an eschatological tradition."""
    node_id: str
    tradition: Tradition
    archetype: EschatologicalArchetype
    title: str
    description: str
    keywords: Set[str]
    estimated_relative_order: int # 1 = early signs, 10 = final resolution


@dataclass
class GeopoliticalEvent:
    """A real-world geopolitical event or intelligence indicator."""
    event_id: str
    timestamp_iso: str
    location: str
    summary: str
    keywords: Set[str]
    severity_score: float # [0.0, 1.0]


@dataclass
class AlignmentReport:
    """Result of correlating a real-world event with eschatological DAG nodes."""
    event_id: str
    top_matching_nodes: List[Tuple[str, float]] # (node_id, affinity_score)
    primary_archetype: EschatologicalArchetype
    weaponization_risk_index: float             # Risk of deliberate narrative priming [0.0, 1.0]
    cross_tradition_resonance: float           # How strongly this event triggers multiple traditions


class EschatologyDAGEngine:
    """Directed Acyclic Graph engine for comparative eschatology ontologies."""

    def __init__(self):
        self.nodes: Dict[str, EschatologyNode] = {}
        self.adjacency: Dict[str, List[str]] = {}
        self.reverse_adjacency: Dict[str, List[str]] = {}
        self._populate_canonical_ontologies()

    def add_node(self, node: EschatologyNode):
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []
        if node.node_id not in self.reverse_adjacency:
            self.reverse_adjacency[node.node_id] = []

    def add_edge(self, u_id: str, v_id: str):
        """Add directed edge u -> v (u precedes/causes v)."""
        if u_id not in self.nodes or v_id not in self.nodes:
            raise KeyError(f"Both nodes must exist: {u_id}, {v_id}")
        if v_id not in self.adjacency[u_id]:
            self.adjacency[u_id].append(v_id)
        if u_id not in self.reverse_adjacency[v_id]:
            self.reverse_adjacency[v_id].append(u_id)

    def is_acyclic(self) -> bool:
        """Check if graph contains no cycles using DFS cycle detection."""
        visited: Dict[str, int] = {nid: 0 for nid in self.nodes} # 0=unvisited, 1=visiting, 2=visited

        def dfs(curr: str) -> bool:
            visited[curr] = 1
            for nxt in self.adjacency.get(curr, []):
                if visited[nxt] == 1:
                    return False # cycle detected
                if visited[nxt] == 0:
                    if not dfs(nxt):
                        return False
            visited[curr] = 2
            return True

        for node_id in self.nodes:
            if visited[node_id] == 0:
                if not dfs(node_id):
                    return False
        return True

    def topological_sort(self) -> List[str]:
        """Return deterministic topological order of narrative nodes via Kahn's algorithm."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for u in self.adjacency:
            for v in self.adjacency[u]:
                in_degree[v] += 1

        queue = sorted([nid for nid in self.nodes if in_degree[nid] == 0])
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for nxt in sorted(self.adjacency.get(curr, [])):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
            queue.sort()

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle; topological sort impossible.")
        return order

    def _populate_canonical_ontologies(self):
        """Initialize the 4 canonical traditions and their causal causal DAG links."""
        canonical_nodes = [
            # --- ISLAMIC TRADITION ---
            EschatologyNode(
                "ISL_01", Tradition.ISLAMIC, EschatologicalArchetype.COSMIC_ENVIRONMENTAL_SIGN,
                "Drying of Euphrates", "River Euphrates uncovers mountain of gold / severe water crisis",
                {"euphrates", "river", "drought", "gold", "water", "iraq", "syria"}, 1
            ),
            EschatologyNode(
                "ISL_02", Tradition.ISLAMIC, EschatologicalArchetype.GEOPOLITICAL_TRIBULATION,
                "Al-Malahim (Great Tribulation)", "Turbulent conflicts across Levant and Sham",
                {"sham", "levant", "conflict", "war", "tribulation", "syria", "jerusalem"}, 3
            ),
            EschatologyNode(
                "ISL_03", Tradition.ISLAMIC, EschatologicalArchetype.MESSIANIC_DELIVERER,
                "Emergence of Al-Mahdi", "Righteous leader establishes global justice after tyranny",
                {"mahdi", "justice", "caliph", "mecca", "medina", "restoration"}, 5
            ),
            EschatologyNode(
                "ISL_04", Tradition.ISLAMIC, EschatologicalArchetype.ARCH_DECEIVER_TYRANT,
                "Manifestation of Al-Masih ad-Dajjal", "False Messiah / Arch-Deceiver exercising occult power",
                {"dajjal", "antichrist", "deceiver", "false_messiah", "miracles", "tyranny"}, 6
            ),
            EschatologyNode(
                "ISL_05", Tradition.ISLAMIC, EschatologicalArchetype.COSMIC_CLIMACTIC_BATTLE,
                "Descent of Isa & Defeat of Dajjal", "Jesus descends at White Minaret in Damascus, defeats Dajjal at Lod",
                {"isa", "jesus", "damascus", "lod", "battle", "victory"}, 8
            ),
            EschatologyNode(
                "ISL_06", Tradition.ISLAMIC, EschatologicalArchetype.GOLDEN_AGE_RESTORATION,
                "Era of Global Peace & Abundance", "Wolf feeds with sheep, divine peace reigns",
                {"peace", "abundance", "justice", "blessing", "harmony"}, 10
            ),

            # --- RABBINIC TRADITION ---
            EschatologyNode(
                "RAB_01", Tradition.RABBINIC, EschatologicalArchetype.MORAL_DECAY_PRECURSOR,
                "Ikvei de-Meshicha (Heels of Messiah)", "Chutzpah increases, truth is absent, generational rebellion",
                {"chutzpah", "insolence", "truth_lacking", "decline", "rebellion"}, 1
            ),
            EschatologyNode(
                "RAB_02", Tradition.RABBINIC, EschatologicalArchetype.GEOPOLITICAL_TRIBULATION,
                "Chevlei Mashiach & Ingathering", "Birthpangs of Messiah, Kibbutz Galuyot (ingathering of exiles)",
                {"ingathering", "exiles", "israel", "jerusalem", "birthpangs", "tribulation"}, 3
            ),
            EschatologyNode(
                "RAB_03", Tradition.RABBINIC, EschatologicalArchetype.COSMIC_CLIMACTIC_BATTLE,
                "War of Gog u'Magog", "Global coalition marches against Jerusalem",
                {"gog", "magog", "coalition", "jerusalem", "war", "nations"}, 6
            ),
            EschatologyNode(
                "RAB_04", Tradition.RABBINIC, EschatologicalArchetype.MESSIANIC_DELIVERER,
                "Mashiach ben David & Third Temple", "Righteous Davidic king, Beit HaMikdash restored",
                {"mashiach", "david", "temple", "jerusalem", "redemption"}, 8
            ),
            EschatologyNode(
                "RAB_05", Tradition.RABBINIC, EschatologicalArchetype.GOLDEN_AGE_RESTORATION,
                "Olam Ha-Ba (World to Come)", "Universal knowledge of God, no war or envy",
                {"olam_haba", "universal_peace", "knowledge", "resurrection"}, 10
            ),

            # --- VEDIC TRADITION ---
            EschatologyNode(
                "VED_01", Tradition.VEDIC, EschatologicalArchetype.MORAL_DECAY_PRECURSOR,
                "Kali Yuga Apex (Adharma Ascendancy)", "Dharma reduced to one quarter, hypocrisy, greed, climatic imbalance",
                {"kali_yuga", "adharma", "hypocrisy", "greed", "decay", "drought"}, 1
            ),
            EschatologyNode(
                "VED_02", Tradition.VEDIC, EschatologicalArchetype.ARCH_DECEIVER_TYRANT,
                "Dominion of Kali Asura", "Corrupt rulers oppress population, deception normalized",
                {"kali_asura", "corrupt_kings", "tyranny", "deception", "oppression"}, 4
            ),
            EschatologyNode(
                "VED_03", Tradition.VEDIC, EschatologicalArchetype.MESSIANIC_DELIVERER,
                "Descent of Kalki Avatar", "Tenth avatar of Vishnu appears on white winged steed Devadatta with blazing sword",
                {"kalki", "vishnu", "avatar", "devadatta", "sword", "shambhala"}, 7
            ),
            EschatologyNode(
                "VED_04", Tradition.VEDIC, EschatologicalArchetype.COSMIC_CLIMACTIC_BATTLE,
                "Annihilation of Adharma", "Decisive cleansing of wicked rulers and demon cohorts",
                {"cleansing", "battle", "destruction_of_evil", "victory", "annihilation"}, 8
            ),
            EschatologyNode(
                "VED_05", Tradition.VEDIC, EschatologicalArchetype.GOLDEN_AGE_RESTORATION,
                "Krita / Satya Yuga Dawn", "Golden age of pure Dharma, truth, long life, and spiritual enlightenment",
                {"satya_yuga", "dharma", "golden_age", "truth", "enlightenment"}, 10
            ),

            # --- BIBLICAL TRADITION ---
            EschatologyNode(
                "BIB_01", Tradition.BIBLICAL, EschatologicalArchetype.MORAL_DECAY_PRECURSOR,
                "Great Apostasy & Signs", "Love of many waxes cold, false prophets, earthquakes, famines",
                {"apostasy", "famine", "earthquake", "rumors_of_war", "signs"}, 1
            ),
            EschatologyNode(
                "BIB_02", Tradition.BIBLICAL, EschatologicalArchetype.ARCH_DECEIVER_TYRANT,
                "Rise of the Antichrist & False Prophet", "Global beast system, mandatory mark, economic control",
                {"antichrist", "beast", "false_prophet", "mark", "surveillance", "deception"}, 5
            ),
            EschatologyNode(
                "BIB_03", Tradition.BIBLICAL, EschatologicalArchetype.COSMIC_CLIMACTIC_BATTLE,
                "Battle of Armageddon", "Kings of the earth gather at Megiddo / Jezreel Valley",
                {"armageddon", "megiddo", "kings", "battle", "winepress", "wrath"}, 7
            ),
            EschatologyNode(
                "BIB_04", Tradition.BIBLICAL, EschatologicalArchetype.MESSIANIC_DELIVERER,
                "Parousia (Second Coming of Christ)", "Triumphant return on white horse, slaying beast with word",
                {"parousia", "second_coming", "christ", "white_horse", "king_of_kings"}, 8
            ),
            EschatologyNode(
                "BIB_05", Tradition.BIBLICAL, EschatologicalArchetype.GOLDEN_AGE_RESTORATION,
                "Millennial Kingdom & New Jerusalem", "1000-year reign of righteousness, death swallowed up",
                {"millennium", "new_jerusalem", "righteousness", "peace", "eternal"}, 10
            ),
        ]

        for n in canonical_nodes:
            self.add_node(n)

        # Build causal edges within each tradition
        # Islamic causal chain
        self.add_edge("ISL_01", "ISL_02")
        self.add_edge("ISL_02", "ISL_03")
        self.add_edge("ISL_03", "ISL_04")
        self.add_edge("ISL_04", "ISL_05")
        self.add_edge("ISL_05", "ISL_06")

        # Rabbinic causal chain
        self.add_edge("RAB_01", "RAB_02")
        self.add_edge("RAB_02", "RAB_03")
        self.add_edge("RAB_03", "RAB_04")
        self.add_edge("RAB_04", "RAB_05")

        # Vedic causal chain
        self.add_edge("VED_01", "VED_02")
        self.add_edge("VED_02", "VED_03")
        self.add_edge("VED_03", "VED_04")
        self.add_edge("VED_04", "VED_05")

        # Biblical causal chain
        self.add_edge("BIB_01", "BIB_02")
        self.add_edge("BIB_02", "BIB_03")
        self.add_edge("BIB_03", "BIB_04")
        self.add_edge("BIB_04", "BIB_05")


class CrossTraditionSemanticAffinity:
    """Computes cross-tradition semantic and archetypal congruence matrices."""

    def __init__(self, dag: EschatologyDAGEngine):
        self.dag = dag

    def compute_archetypal_affinity_matrix(self) -> Dict[Tuple[Tradition, Tradition], float]:
        """
        Compute pairwise semantic congruence between traditions based on shared archetypes & keywords.
        Congruence S(T1, T2) in [0.0, 1.0].
        """
        traditions = list(Tradition)
        affinity_matrix = {}

        for t1 in traditions:
            for t2 in traditions:
                if t1 == t2:
                    affinity_matrix[(t1, t2)] = 1.0
                    continue

                nodes_t1 = [n for n in self.dag.nodes.values() if n.tradition == t1]
                nodes_t2 = [n for n in self.dag.nodes.values() if n.tradition == t2]

                # Match by archetype
                shared_archetypes = 0
                archetypes_t1 = {n.archetype for n in nodes_t1}
                archetypes_t2 = {n.archetype for n in nodes_t2}
                jaccard_archetype = len(archetypes_t1 & archetypes_t2) / max(1, len(archetypes_t1 | archetypes_t2))

                # Match by keywords Jaccard
                kw_t1 = set.union(*[n.keywords for n in nodes_t1])
                kw_t2 = set.union(*[n.keywords for n in nodes_t2])
                jaccard_kw = len(kw_t1 & kw_t2) / max(1, len(kw_t1 | kw_t2))

                combined_affinity = float(0.70 * jaccard_archetype + 0.30 * jaccard_kw)
                affinity_matrix[(t1, t2)] = combined_affinity

        return affinity_matrix


class GeopoliticalEschatologyCorrelator:
    """Correlates real-world geopolitical events with eschatological milestone nodes."""

    def __init__(self, dag_engine: Optional[EschatologyDAGEngine] = None):
        self.dag = dag_engine or EschatologyDAGEngine()

    def correlate_event(self, event: GeopoliticalEvent) -> AlignmentReport:
        """
        Evaluate semantic and keyword alignment between a geopolitical intelligence event
        and all eschatological DAG nodes.
        """
        node_scores: List[Tuple[str, float]] = []
        traditions_hit: Dict[Tradition, float] = {t: 0.0 for t in Tradition}

        event_text_words = set(re.findall(r'\b\w+\b', event.summary.lower())) | event.keywords

        for node_id, node in self.dag.nodes.items():
            # Keyword overlap (Jaccard similarity)
            overlap = len(node.keywords & event_text_words)
            union = len(node.keywords | event_text_words)
            jaccard = (overlap / union) if union > 0 else 0.0

            # Direct keyword hits boost
            hit_score = overlap * 0.25
            total_affinity = min(1.0, jaccard * 2.0 + hit_score)

            if total_affinity > 0.05:
                node_scores.append((node_id, float(total_affinity)))
                traditions_hit[node.tradition] = max(traditions_hit[node.tradition], float(total_affinity))

        # Sort matches by affinity
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = node_scores[:5]

        # Determine primary archetype
        if top_matches:
            top_node = self.dag.nodes[top_matches[0][0]]
            primary_archetype = top_node.archetype
        else:
            primary_archetype = EschatologicalArchetype.GEOPOLITICAL_TRIBULATION

        # Cross-tradition resonance: How many distinct traditions have affinity > 0.2
        active_traditions = sum(1 for score in traditions_hit.values() if score >= 0.2)
        cross_resonance = float(active_traditions / len(Tradition))

        # Narrative Weaponization Risk Index:
        # High when event severity is high AND cross-tradition resonance is high AND targets climactic/deceiver nodes
        climactic_multiplier = 1.3 if primary_archetype in (
            EschatologicalArchetype.ARCH_DECEIVER_TYRANT,
            EschatologicalArchetype.COSMIC_CLIMACTIC_BATTLE,
            EschatologicalArchetype.MESSIANIC_DELIVERER
        ) else 0.9

        top_affinity = top_matches[0][1] if top_matches else 0.0
        weaponization_risk = float(np.clip(
            (event.severity_score * 0.4 + top_affinity * 0.35 + cross_resonance * 0.25) * climactic_multiplier,
            0.0, 1.0
        ))

        return AlignmentReport(
            event_id=event.event_id,
            top_matching_nodes=top_matches,
            primary_archetype=primary_archetype,
            weaponization_risk_index=weaponization_risk,
            cross_tradition_resonance=cross_resonance
        )
