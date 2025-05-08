import unittest
from sin_ai.core.emotions import EmotionEngine, EmotionResponse

class TestEmotionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EmotionEngine()
    
    def test_detection(self):
        response = self.engine.detect_emotion("Я так рад этому!")
        self.assertEqual(response.emotion, "happy")
        
    def test_state_update(self):
        self.engine.update_state("happy")
        self.assertEqual(self.engine.state, "happy")
        
    def test_response_generation(self):
        response = self.engine.generate_response("happy")
        self.assertIn(response, ["Отлично!", "Я рад за вас!", "Прекрасный день!"])
