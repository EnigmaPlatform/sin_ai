import logging
from typing import Dict

logger = logging.getLogger(__name__)

class PersonalityCore:
    MODES = {
        'neutral': {"traits": [], "responses": ["Я вас слушаю", "Продолжайте"]},
        'scientist': {
            "traits": ["аналитичный", "точный"],
            "responses": ["Согласно моим данным...", "Анализ показывает..."]
        },
        'friendly': {
            "traits": ["дружелюбный", "эмпатичный"],
            "responses": ["Рад вас слышать!", "Как я могу помочь?"]
        }
    }

    def __init__(self):
        self.current_mode = 'neutral'
        self.custom_responses = []

    def set_mode(self, mode: str):
        if mode in self.MODES:
            self.current_mode = mode
            logger.info(f"Personality set to {mode} mode")
        else:
            logger.warning(f"Unknown personality mode: {mode}")

    def get_response(self, message: str) -> str:
        import random
        base_responses = self.MODES[self.current_mode]["responses"]
        responses = base_responses + self.custom_responses
        
        if not responses:
            return "Я вас слушаю"
            
        return random.choice(responses)
