"""Course 153: Linux Kernel Slab Memory Cache Allocator Simulator"""
class KernelSlabCache:
    def __init__(self, obj_size: int = 32, slab_capacity: int = 16):
        self.obj_size = obj_size
        self.capacity = slab_capacity
        self.free_list = [bytearray(obj_size) for _ in range(slab_capacity)]
        self.allocated = set()

    def kmalloc(self) -> bytearray:
        if not self.free_list:
            raise MemoryError("Slab cache exhausted")
        buf = self.free_list.pop()
        self.allocated.add(id(buf))
        return buf

    def kfree(self, buf: bytearray):
        if id(buf) in self.allocated:
            self.allocated.remove(id(buf))
            self.free_list.append(buf)
