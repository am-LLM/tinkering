"""Course 152: Linux Flattened Device Tree (DTS) Node & Register Validator"""
import re

class DeviceTreeParser:
    @staticmethod
    def parse_reg_property(node_text: str) -> list:
        # Match reg = <0xaddr 0xsize>
        matches = re.findall(r'reg\s*=\s*<([^>]+)>', node_text)
        regs = []
        for m in matches:
            tokens = m.strip().split()
            for i in range(0, len(tokens), 2):
                if i + 1 < len(tokens):
                    addr = int(tokens[i], 16) if tokens[i].startswith("0x") else int(tokens[i])
                    size = int(tokens[i+1], 16) if tokens[i+1].startswith("0x") else int(tokens[i+1])
                    regs.append((addr, size))
        return regs
