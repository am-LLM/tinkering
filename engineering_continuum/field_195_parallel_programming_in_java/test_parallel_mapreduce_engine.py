from parallel_mapreduce_engine import ParallelMapReduceEngine

def test_mapreduce():
    data = [1, 2, 3, 4, 5]
    res = ParallelMapReduceEngine.map_reduce(data, lambda x: x*2, lambda a, b: a+b)
    assert res == 30
