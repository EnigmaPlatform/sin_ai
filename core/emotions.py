import random
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class EmotionResponse:
    """Структура для хранения ответа эмоционального движка."""
    emotion: str
    confidence: float
    triggers: List[str]

class EmotionEngine:
    def __init__(self):
        self.state = 'neutral'
        self._emotion_model = self._load_default_model()
    
    def _load_default_model(self) -> Dict[str, Dict]:
        """Базовая модель для детекции эмоций (можно заменить на ML-модель)."""
        return {
            "happy": {
                "keywords": ["рад", "отлично", "ура", "счастье", "люблю"],
                "response_prob": 0.8,
            },
            "sad": {
                "keywords": ["грустно", "плохо", "тоска", "слезы"],
                "response_prob": 0.7,
            },
            "angry": {
                "keywords": ["злой", "бесит", "ненавижу", "раздражает"],
                "response_prob": 0.9,
            },
            "neutral": {
                "keywords": [],
                "response_prob": 0.5,
            }
        }

    def detect_emotion(self, text: str) -> EmotionResponse:
        """Определяет эмоцию в тексте и возвращает структурированный ответ."""
        if not text:
            return EmotionResponse(emotion="neutral", confidence=1.0, triggers=[])
        
        text_lower = text.lower()
        detected_emotions = []
        
        for emotion, data in self._emotion_model.items():
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    detected_emotions.append((emotion, data["response_prob"]))
                    break
        
        if not detected_emotions:
            return EmotionResponse(emotion="neutral", confidence=0.5, triggers=[])
        
        # Выбираем эмоцию с наибольшей вероятностью
        best_emotion, confidence = max(detected_emotions, key=lambda x: x[1])
        triggers = [kw for kw in self._emotion_model[best_emotion]["keywords"] if kw in text_lower]
        
        return EmotionResponse(
            emotion=best_emotion,
            confidence=confidence,
            triggers=triggers,
        )
    
    def update_state(self, new_emotion: str) -> None:
        """Обновляет текущее эмоциональное состояние."""
        valid_emotions = list(self._emotion_model.keys())
        if new_emotion not in valid_emotions:
            raise ValueError(f"Неизвестная эмоция. Допустимые: {valid_emotions}")
        self.state = new_emotion
    
    def generate_response(self, emotion: Optional[str] = None) -> str:
        """Генерирует ответ в зависимости от эмоции."""
        emotion = emotion or self.state
        responses = {
            "happy": ["Отлично!", "Я рад за вас!", "Прекрасный день!"],
            "sad": ["Мне жаль...", "Может, обсудим?", "Всё наладится."],
            "angry": ["Попробуйте успокоиться.", "Дышите глубже.", "Я вас понимаю."],
            "neutral": ["Хмм...", "Интересно.", "Продолжайте."],
        }
        return random.choice(responses.get(emotion, ["Я вас слушаю."]))
