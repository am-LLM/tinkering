"""Course 193: Token Bucket Rate Limiter & Packet Traffic Shaper"""
import time

class TokenBucketShaper:
    def __init__(self, rate_tokens_per_sec: float = 100.0, burst_capacity: float = 500.0):
        self.rate = rate_tokens_per_sec
        self.capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_update = 0.0

    def consume(self, packet_bytes: float, current_time: float) -> bool:
        elapsed = current_time - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = current_time
        
        if self.tokens >= packet_bytes:
            self.tokens -= packet_bytes
            return True
        return False
