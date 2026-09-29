from fdtd_schrodinger_wavepacket import SchrodingerFDTD
def test_wavepacket_propagation():
    solver = SchrodingerFDTD()
    x_pos = [solver.step() for _ in range(50)]
    assert x_pos[-1] > x_pos[0], "Expected positive wave packet drift"
