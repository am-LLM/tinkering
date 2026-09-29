"""Course 131: Variant Call Format (VCF) Parser & Variant Impact Annotator"""
class VCFVariantAnnotator:
    @staticmethod
    def parse_vcf_line(line: str) -> dict:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            return {}
        chrom, pos, id_, ref, alt = parts[:5]
        var_type = "SNV" if len(ref) == 1 and len(alt) == 1 else "INDEL"
        return {"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt, "type": var_type}
