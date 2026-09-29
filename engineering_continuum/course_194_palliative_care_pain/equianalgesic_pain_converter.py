"""Course 194: Equianalgesic Opioid Dose Converter (Oral Morphine Milligram Equivalents)"""
class EquianalgesicPainConverter:
    # Multipliers to convert dose in mg to oral MME
    CONVERSION_FACTORS = {
        "oral_morphine": 1.0,
        "oral_oxycodone": 1.5,
        "oral_hydromorphone": 4.0,
        "iv_morphine": 3.0,
        "oral_codeine": 0.15
    }

    @classmethod
    def calculate_mme(cls, drug_type: str, dose_mg: float) -> float:
        factor = cls.CONVERSION_FACTORS.get(drug_type.lower())
        if factor is None:
            raise ValueError(f"Unknown drug type: {drug_type}")
        return float(dose_mg * factor)
