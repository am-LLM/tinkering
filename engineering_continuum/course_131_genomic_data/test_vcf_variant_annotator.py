from vcf_variant_annotator import VCFVariantAnnotator

def test_vcf_parsing():
    line = "chr1\t12345\trs1\tA\tG\t100\tPASS"
    v = VCFVariantAnnotator.parse_vcf_line(line)
    assert v["type"] == "SNV"
    assert v["ref"] == "A"
