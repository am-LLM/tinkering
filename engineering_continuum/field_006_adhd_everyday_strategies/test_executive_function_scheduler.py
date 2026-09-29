import unittest
from executive_function_scheduler import ExecutiveFunctionScheduler

class TestScheduler(unittest.TestCase):
    def test_chunking(self):
        s = ExecutiveFunctionScheduler()
        chunks = s.chunk_tasks([{"name": "Read", "duration": 60}])
        self.assertEqual(len(chunks), 3)
