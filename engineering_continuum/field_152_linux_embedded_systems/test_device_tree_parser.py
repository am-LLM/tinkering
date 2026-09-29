from device_tree_parser import DeviceTreeParser

def test_dts_parser():
    dts = 'uart0: serial@10000000 { reg = <0x10000000 0x100>; };'
    regs = DeviceTreeParser.parse_reg_property(dts)
    assert regs == [(0x10000000, 0x100)]
