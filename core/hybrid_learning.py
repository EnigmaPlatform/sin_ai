import requests
from typing import Tuple

class HybridLearner:
    def __init__(self, local_model, api_endpoint: str = "https://api.deepseek.com/v1"):
        self.local_model = local_model
        self.api_endpoint = api_endpoint
        self.confidence_threshold = 0.65
    
    def train_with_feedback(self, prompt: str, user_feedback: int) -> Tuple[str, str]:
        """Обучение с подкреплением на основе оценок"""
        try:
            # Локальная генерация
            local_response = self.local_model.generate(prompt)
            local_confidence = self._calculate_confidence(local_response)
            
            # Динамический выбор источника
            if local_confidence < self.confidence_threshold:
                api_response = self._query_deepseek(prompt)
                self._adjust_weights(prompt, api_response, user_feedback)
                return api_response, "api"
            
            self._adjust_weights(prompt, local_response, user_feedback)
            return local_response, "local"
        except APIError as e:
            self._handle_api_error(e)
            return local_response, "local_fallback"

    def _adjust_weights(self, prompt: str, response: str, rating: int):
        """Алгоритм корректировки весов"""
        # RLHF-логика
        adjustment = 0.1 * (rating - 3)  # Центрирование вокруг 3
        
        # Обновление параметров
        self.local_model.update_embedding_weights(prompt, adjustment)
        self.local_model.dialogue_reward += adjustment * 0.5
        
        # Логирование для последующего анализа
        self._log_training_example(prompt, response, rating)

    def _query_deepseek(self, prompt: str) -> str:
        """Интеллектуальный запрос к API"""
        payload = {
            "prompt": prompt,
            "context": self.local_model.get_context(),
            "mode": "teaching"  # Специальный режим для обучения
        }
        
        response = requests.post(
            f"{self.api_endpoint}/chat",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise APIError(f"API request failed: {response.status_code}")
