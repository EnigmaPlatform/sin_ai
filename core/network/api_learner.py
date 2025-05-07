# Интеграция с DeepSeek
import requests

class DeepSeekAPI:
    @staticmethod
    def query(prompt):
        response = requests.post(
            "https://api.deepseek.com/v1/chat",
            headers={"Authorization": "Bearer YOUR_KEY"},
            json={"prompt": prompt}
        )
        return response.json()
