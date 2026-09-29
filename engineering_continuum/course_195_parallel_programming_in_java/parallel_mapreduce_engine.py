"""Course 195: Parallel MapReduce Chunk Partition & Reduce Framework"""
from functools import reduce

class ParallelMapReduceEngine:
    @staticmethod
    def map_reduce(items: list, map_fn, reduce_fn, num_partitions: int = 4):
        chunk_size = max(1, len(items) // num_partitions)
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        
        mapped_chunks = [[map_fn(x) for x in chunk] for chunk in chunks]
        flat_mapped = [item for sublist in mapped_chunks for item in sublist]
        
        if not flat_mapped:
            return None
        return reduce(reduce_fn, flat_mapped)
