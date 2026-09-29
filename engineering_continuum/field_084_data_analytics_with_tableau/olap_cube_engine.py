"""Course 084: Multidimensional OLAP Data Cube Rollup & Aggregator"""
from typing import List, Dict, Any

class OLAPDataCube:
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def aggregate(self, group_by_dims: List[str], metric: str, agg_func=sum) -> Dict[tuple, float]:
        cube = {}
        for r in self.records:
            key = tuple(r[dim] for dim in group_by_dims)
            cube.setdefault(key, []).append(r[metric])
        return {k: float(agg_func(vals)) for k, vals in cube.items()}
