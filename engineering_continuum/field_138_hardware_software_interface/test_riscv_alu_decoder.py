from riscv_alu_decoder import RISCVALUDecoder

def test_riscv_alu():
    assert RISCVALUDecoder.execute_alu("ADD", 5, 10) == 15
    assert RISCVALUDecoder.execute_alu("XOR", 0xFF, 0x0F) == 0xF0
