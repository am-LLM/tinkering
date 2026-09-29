from debruijn_graph_assembler import DeBruijnAssembler

def test_debruijn():
    asm = DeBruijnAssembler(k=3)
    asm.build_graph(["ATG", "TGC", "GCC"])
    seq = asm.assemble_simple_path("AT")
    assert seq == "ATGCC"
