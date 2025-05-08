import unittest
from sin_ai.core.memory import MemorySystem

class TestMemorySystem(unittest.TestCase):
    def test_add_retrieve_memory(self):
        memory = MemorySystem()
        memory.add_memory("Test content", ["test"])
        results = memory.retrieve_memory("test")
        self.assertGreater(len(results), 0)
