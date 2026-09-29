"""Course 141: Shannon Entropy & Huffman Coding Prefix Tree Codec"""
import math
import heapq

class ShannonHuffmanCodec:
    @staticmethod
    def shannon_entropy(probabilities: list) -> float:
        return float(-sum(p * math.log2(p) for p in probabilities if p > 0))

    @staticmethod
    def huffman_codes(frequencies: dict) -> dict:
        heap = [[weight, [sym, ""]] for sym, weight in frequencies.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return {item[0]: item[1] for item in heap[0][1:]}
