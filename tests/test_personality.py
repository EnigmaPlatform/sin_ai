import unittest
from sin_ai.core.personality import PersonalityCore

class TestPersonalityCore(unittest.TestCase):
    def setUp(self):
        self.personality = PersonalityCore()
    
    def test_archetype_change(self):
        self.personality.set_archetype("scientist")
        self.assertEqual(self.personality.current_archetype, "scientist")
    
    def test_format_response(self):
        response = self.personality.format_response("Test")
        self.assertEqual(response, "Test")  # Для neutral архетипа
