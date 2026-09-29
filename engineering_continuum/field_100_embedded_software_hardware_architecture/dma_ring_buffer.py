"""Course 100: Memory-Mapped DMA Circular Ring Buffer with Atomic Pointers"""
class DMARingBuffer:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.buffer = [0] * capacity
        self.head = 0
        self.tail = 0
        self.count = 0

    def write_dma(self, byte_val: int) -> bool:
        if self.count >= self.capacity:
            return False # Overflow
        self.buffer[self.head] = byte_val & 0xFF
        self.head = (self.head + 1) % self.capacity
        self.count += 1
        return True

    def read_cpu(self) -> int:
        if self.count == 0:
            return -1 # Underflow
        val = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.capacity
        self.count -= 1
        return val
