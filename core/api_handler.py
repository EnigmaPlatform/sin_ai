# core/api_handler.py

import os
import requests
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
from core.api_cache import APICache
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

load_dotenv()

class DeepSeekAPIHandler:
   def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            logger.error("DeepSeek API key not found in environment variables")
            raise ValueError("API key is required. Set DEEPSEEK_API_KEY in .env file")
    
        self.base_url = "https://api.deepseek.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
    })
    
    def query(self, prompt: str, context: Optional[List[str]] = None) -> Dict:
        cache_key = f"{prompt}-{'-'.join(context or [])}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")
        """Отправка запроса к DeepSeek API"""
        if context is None:
            context = []
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful AI assistant that provides detailed explanations and code examples.'
                },
                *[{'role': 'user', 'content': msg} for msg in context],
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload)
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_response(data)
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {'error': str(e)}
    
    def _parse_response(self, data: Dict) -> Dict:
        """Парсинг ответа API"""
        try:
            content = data['choices'][0]['message']['content']
            
            # Попытка разобрать JSON, если ответ в таком формате
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Стандартный текстовый ответ
            return {
                'response': content,
                'timestamp': datetime.now().isoformat()
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {e}")
            return {'error': 'Invalid API response format'}
