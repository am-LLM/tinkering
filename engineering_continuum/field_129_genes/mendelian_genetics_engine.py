"""Course 129: Hardy-Weinberg Allele Equilibrium & Punnett Square Engine"""
class MendelianGeneticsEngine:
    @staticmethod
    def hardy_weinberg(p: float) -> dict:
        q = 1.0 - p
        return {
            "p": p,
            "q": q,
            "p2_homozygous_dom": p ** 2,
            "2pq_heterozygous": 2 * p * q,
            "q2_homozygous_rec": q ** 2
        }

    @staticmethod
    def monohybrid_cross(parent1: str, parent2: str) -> dict:
        outcomes = [a + b for a in parent1 for b in parent2]
        counts = {}
        for o in outcomes:
            norm = "".join(sorted(o))
            counts[norm] = counts.get(norm, 0) + 1
        return {k: v / len(outcomes) for k, v in counts.items()}
