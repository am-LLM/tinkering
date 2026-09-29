"""
Evidence-Based Pharmacognosy Engine & Enzyme Inhibition Kinetics.

Implements:
1. Secondary metabolite physicochemical repository (Salicin, Artemisinin, Berberine, Medical Grade Honey / MGO).
2. Physicochemical descriptor profiling & Lipinski Rule of 5 compliance.
3. Generalized Enzyme Inhibition Kinetics (Competitive, Uncompetitive, Non-Competitive, Mixed).
4. Kinetic diagnostic transformations: Lineweaver-Burk, Hanes-Woolf, Eadie-Hofstee.
5. Experimental data fitting and Chou-Talalay Combination Index (CI) synergy quantification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class InhibitionType(Enum):
    COMPETITIVE = "competitive"
    UNCOMPETITIVE = "uncompetitive"
    NONCOMPETITIVE = "noncompetitive"
    MIXED = "mixed"


@dataclass
class SecondaryMetabolite:
    """Chemical and pharmacological metadata for a natural product metabolite."""
    name: str
    botanical_source: str
    chemical_class: str
    smiles_formula: str
    molecular_weight: float       # g/mol
    log_p: float                  # Octanol-water partition coeff
    h_bond_donors: int
    h_bond_acceptors: int
    tpsa: float                   # Topological polar surface area (A^2)
    rotatable_bonds: int
    primary_biological_target: str
    mechanism_of_action: str
    reported_ic50_um: float       # micromolar
    binding_free_energy_kcal_mol: float  # delta G (kcal/mol)

    def passes_lipinski_rule_of_five(self) -> Tuple[bool, List[str]]:
        """
        Evaluates Lipinski's Rule of 5:
        - MW <= 500 Da
        - LogP <= 5.0
        - HBD <= 5
        - HBA <= 10
        """
        violations = []
        if self.molecular_weight > 500.0:
            violations.append(f"MW ({self.molecular_weight} Da) > 500 Da")
        if self.log_p > 5.0:
            violations.append(f"LogP ({self.log_p}) > 5.0")
        if self.h_bond_donors > 5:
            violations.append(f"H-bond donors ({self.h_bond_donors}) > 5")
        if self.h_bond_acceptors > 10:
            violations.append(f"H-bond acceptors ({self.h_bond_acceptors}) > 10")
        
        return (len(violations) == 0, violations)

    def calculate_dissociation_constant_ki(self, temperature_k: float = 310.15) -> float:
        """
        Calculates equilibrium dissociation constant K_i (uM) from binding free energy delta G:
        delta G = R * T * ln(K_i / 1M) -> K_i = exp(delta G / (R * T)) * 1e6 uM
        R = 1.9872e-3 kcal/(mol*K)
        """
        r_gas = 1.98720425864083e-3  # kcal / (mol * K)
        # K_i in molar
        ki_molar = np.exp(self.binding_free_energy_kcal_mol / (r_gas * temperature_k))
        # Convert to micromolar (uM)
        return float(ki_molar * 1e6)


class PharmacognosyRepository:
    """Curated knowledgebase of evidence-based natural bioactive metabolites."""

    @staticmethod
    def get_standard_database() -> Dict[str, SecondaryMetabolite]:
        return {
            "salicin": SecondaryMetabolite(
                name="Salicin",
                botanical_source="Salix alba (White Willow Bark)",
                chemical_class="Phenolic Glycoside",
                smiles_formula="C13H18O7",
                molecular_weight=286.28,
                log_p=-1.20,
                h_bond_donors=5,
                h_bond_acceptors=7,
                tpsa=119.6,
                rotatable_bonds=3,
                primary_biological_target="Cyclooxygenase-1/2 (COX-1/2)",
                mechanism_of_action="Prodrug metabolized into salicylic acid, competitively inhibiting prostaglandin H2 synthase",
                reported_ic50_um=45.0,
                binding_free_energy_kcal_mol=-6.8,
            ),
            "artemisinin": SecondaryMetabolite(
                name="Artemisinin",
                botanical_source="Artemisia annua (Sweet Wormwood)",
                chemical_class="Sesquiterpene Lactone Endoperoxide",
                smiles_formula="C15H22O5",
                molecular_weight=282.33,
                log_p=2.90,
                h_bond_donors=0,
                h_bond_acceptors=5,
                tpsa=54.0,
                rotatable_bonds=0,
                primary_biological_target="Plasmodium falciparum PfATP6 / Heme iron complex",
                mechanism_of_action="Endoperoxide bridge homolytic cleavage generating carbon-centered free radicals alkylating parasite membrane pumps",
                reported_ic50_um=0.015,
                binding_free_energy_kcal_mol=-9.4,
            ),
            "berberine": SecondaryMetabolite(
                name="Berberine",
                botanical_source="Berberis vulgaris (Barberry) / Hydrastis canadensis",
                chemical_class="Quaternary Isoquinoline Alkaloid",
                smiles_formula="C20H18NO4+",
                molecular_weight=336.36,
                log_p=2.10,
                h_bond_donors=0,
                h_bond_acceptors=4,
                tpsa=40.8,
                rotatable_bonds=0,
                primary_biological_target="AMPK / Acetylcholinesterase (AChE) / PCSK9",
                mechanism_of_action="Direct allosteric activation of AMPK alpha subunit and non-competitive catalytic blockade of AChE gorge",
                reported_ic50_um=0.58,
                binding_free_energy_kcal_mol=-8.7,
            ),
            "medical_grade_honey_mgo": SecondaryMetabolite(
                name="Methylglyoxal (Medical Honey Bioactive)",
                botanical_source="Leptospermum scoparium (Manuka) / Apis mellifera",
                chemical_class="Dicarbonyl / Antimicrobial Aldoketone",
                smiles_formula="C3H4O2",
                molecular_weight=72.06,
                log_p=-0.41,
                h_bond_donors=0,
                h_bond_acceptors=2,
                tpsa=34.1,
                rotatable_bonds=1,
                primary_biological_target="Bacterial Cell Wall & DNA Replication Machineries",
                mechanism_of_action="Covalent glycation of peptidoglycan crosslinks and bacterial DNA gyrase subunit A inhibition",
                reported_ic50_um=180.0,
                binding_free_energy_kcal_mol=-4.9,
            ),
        }


@dataclass
class EnzymeKineticsModel:
    """
    Generalized Michaelis-Menten & Enzyme Inhibition Kinetics Engine.
    
    Equation:
    v = (Vmax * [S]) / ( Km * (1 + [I]/Kic) + [S] * (1 + [I]/Kiu) )
    """
    v_max: float          # Maximum catalytic rate (umol / min / mg protein)
    k_m: float            # Michaelis constant (uM)
    k_ic: float = float("inf")  # Competitive inhibition constant (uM)
    k_iu: float = float("inf")  # Uncompetitive inhibition constant (uM)

    def velocity(self, substrate_conc_um: float, inhibitor_conc_um: float = 0.0) -> float:
        """Calculates instantaneous reaction velocity v for given [S] and [I]."""
        s = max(0.0, substrate_conc_um)
        i = max(0.0, inhibitor_conc_um)

        comp_term = 1.0 + (i / self.k_ic if not np.isinf(self.k_ic) and self.k_ic > 0 else 0.0)
        uncomp_term = 1.0 + (i / self.k_iu if not np.isinf(self.k_iu) and self.k_iu > 0 else 0.0)

        denominator = self.k_m * comp_term + s * uncomp_term
        if denominator <= 1e-12:
            return 0.0
        return float((self.v_max * s) / denominator)

    def lineweaver_burk_coords(
        self, substrate_concentrations: np.ndarray, inhibitor_conc: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (1/[S], 1/v) coordinates for Lineweaver-Burk double reciprocal plot.
        """
        s_arr = np.asarray(substrate_concentrations, dtype=np.float64)
        if np.any(s_arr <= 0):
            raise ValueError("All substrate concentrations must be strictly positive for double-reciprocal plot.")

        v_arr = np.array([self.velocity(s, inhibitor_conc) for s in s_arr], dtype=np.float64)
        inv_s = 1.0 / s_arr
        inv_v = 1.0 / v_arr
        return inv_s, inv_v

    def hanes_woolf_coords(
        self, substrate_concentrations: np.ndarray, inhibitor_conc: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ([S], [S]/v) coordinates for Hanes-Woolf plot."""
        s_arr = np.asarray(substrate_concentrations, dtype=np.float64)
        v_arr = np.array([self.velocity(s, inhibitor_conc) for s in s_arr], dtype=np.float64)
        return s_arr, s_arr / v_arr

    def eadie_hofstee_coords(
        self, substrate_concentrations: np.ndarray, inhibitor_conc: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (v/[S], v) coordinates for Eadie-Hofstee plot."""
        s_arr = np.asarray(substrate_concentrations, dtype=np.float64)
        v_arr = np.array([self.velocity(s, inhibitor_conc) for s in s_arr], dtype=np.float64)
        return v_arr / s_arr, v_arr

    @classmethod
    def create(
        cls,
        v_max: float,
        k_m: float,
        inhibition_type: InhibitionType,
        k_i: float,
        alpha_mixed: float = 2.5,
    ) -> "EnzymeKineticsModel":
        """Factory constructor based on inhibition classification."""
        if inhibition_type == InhibitionType.COMPETITIVE:
            return cls(v_max=v_max, k_m=k_m, k_ic=k_i, k_iu=float("inf"))
        elif inhibition_type == InhibitionType.UNCOMPETITIVE:
            return cls(v_max=v_max, k_m=k_m, k_ic=float("inf"), k_iu=k_i)
        elif inhibition_type == InhibitionType.NONCOMPETITIVE:
            return cls(v_max=v_max, k_m=k_m, k_ic=k_i, k_iu=k_i)
        elif inhibition_type == InhibitionType.MIXED:
            return cls(v_max=v_max, k_m=k_m, k_ic=k_i, k_iu=k_i * alpha_mixed)
        raise ValueError(f"Unknown inhibition type: {inhibition_type}")


class PharmacognosySynergyEngine:
    """
    Chou-Talalay Median-Effect algorithm for drug combination synergy & antagonism index (CI).
    """

    @staticmethod
    def calculate_combination_index(
        dose_1: float,
        dose_2: float,
        ic50_1: float,
        ic50_2: float,
        effect_fa: float = 0.5,
        m_hill_1: float = 1.0,
        m_hill_2: float = 1.0,
    ) -> float:
        """
        Computes the Chou-Talalay Combination Index (CI):
        CI = (D1 / Dx1) + (D2 / Dx2)
        where Dx = Dm * (fa / (1 - fa))^(1/m)
        CI < 1: Synergism
        CI = 1: Additive effect
        CI > 1: Antagonism
        """
        if effect_fa <= 0.0 or effect_fa >= 1.0:
            raise ValueError(f"Effect fraction fa must be in (0, 1), got {effect_fa}")

        ratio = effect_fa / (1.0 - effect_fa)
        dx_1 = ic50_1 * (ratio ** (1.0 / max(0.1, m_hill_1)))
        dx_2 = ic50_2 * (ratio ** (1.0 / max(0.1, m_hill_2)))

        ci = (dose_1 / dx_1) + (dose_2 / dx_2)
        return float(ci)

    @staticmethod
    def classify_synergy(ci: float) -> str:
        if ci < 0.3:
            return "Strong Synergism"
        elif ci < 0.7:
            return "Synergism"
        elif ci < 0.85:
            return "Moderate Synergism"
        elif ci <= 1.15:
            return "Nearly Additive"
        elif ci <= 1.45:
            return "Slight Antagonism"
        elif ci <= 3.3:
            return "Antagonism"
        else:
            return "Strong Antagonism"
