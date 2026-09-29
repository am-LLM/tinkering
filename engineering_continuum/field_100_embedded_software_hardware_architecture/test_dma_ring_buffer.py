from dma_ring_buffer import DMARingBuffer

def test_ring_buffer():
    rb = DMARingBuffer(capacity=4)
    assert rb.write_dma(0xAA) is True
    assert rb.write_dma(0xBB) is True
    assert rb.read_cpu() == 0xAA
    assert rb.read_cpu() == 0xBB
    assert rb.read_cpu() == -1
