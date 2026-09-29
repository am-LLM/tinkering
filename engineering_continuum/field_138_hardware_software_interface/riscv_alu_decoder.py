"""Course 138: 32-bit RISC-V Instruction ALU & Opcode Decoder Simulator"""
class RISCVALUDecoder:
    @staticmethod
    def execute_alu(op: str, rs1: int, rs2: int) -> int:
        if op == "ADD":
            return (rs1 + rs2) & 0xFFFFFFFF
        elif op == "SUB":
            return (rs1 - rs2) & 0xFFFFFFFF
        elif op == "AND":
            return (rs1 & rs2) & 0xFFFFFFFF
        elif op == "OR":
            return (rs1 | rs2) & 0xFFFFFFFF
        elif op == "XOR":
            return (rs1 ^ rs2) & 0xFFFFFFFF
        elif op == "SLL":
            return (rs1 << (rs2 & 0x1F)) & 0xFFFFFFFF
        return 0
