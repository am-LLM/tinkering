from kernel_slab_allocator import KernelSlabCache

def test_slab():
    slab = KernelSlabCache(32, 2)
    b1 = slab.kmalloc()
    b2 = slab.kmalloc()
    assert len(slab.free_list) == 0
    slab.kfree(b1)
    assert len(slab.free_list) == 1
