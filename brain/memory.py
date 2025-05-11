import json
import numpy as np
from collections import deque
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
import logging
logger = logging.getLogger(__name__)

class SinMemory:
    def __init__(self, max_context: int = 5):
        self.context = deque(maxlen=max_context)
        self.knowledge_graph = []
        self.long_term = []
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_cache = {}
        self.logger = logging.getLogger(__name__)

    def add_dialogue(self, dialogue: dict) -> None:
        """Добавление диалога в граф знаний"""
        entry = {
            'query': dialogue.get('user_query', ''),
            'responses': [],
            'metadata': {
                'has_meta': False,
                'difficulty_distribution': {},
                'emotion_distribution': {}
            }
        }
        
        if 'category' in dialogue:
            entry['category'] = dialogue['category']
        
        for response in dialogue.get('responses', []):
            if isinstance(response, dict):
                text = response.get('text', '')
                entry['responses'].append(text)
                
                if 'meta' in response:
                    entry['metadata']['has_meta'] = True
                    meta = response['meta']
                    if 'difficulty' in meta:
                        entry['metadata']['difficulty_distribution'][meta['difficulty']] = \
                            entry['metadata']['difficulty_distribution'].get(meta['difficulty'], 0) + 1
                    if 'emotion' in meta:
                        entry['metadata']['emotion_distribution'][meta['emotion']] = \
                            entry['metadata']['emotion_distribution'].get(meta['emotion'], 0) + 1
        
        self.knowledge_graph.append(entry)

    def add_interaction(self, user_text: str, ai_text: str) -> None:
        """Добавление взаимодействия в контекст"""
        self.context.append(f"User: {user_text}")
        self.context.append(f"Sin: {ai_text}")
        self._evaluate_importance(user_text)

    def _evaluate_importance(self, text: str) -> None:
        """Оценка важности сообщения"""
        importance = 0.3
        if len(text.split()) > 7:
            importance += 0.2
        if '?' in text:
            importance += 0.1
        if any(word in text.lower() for word in ['важно', 'запомни']):
            importance += 0.4
            
        if importance > 0.5:
            self.remember(text, importance)

    def remember(self, text: str, importance: float = 0.5) -> None:
        """Сохранение важной информации в долговременную память"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self._get_embedding(text)
        
        self.long_term.append({
            "text": text,
            "embedding": self.embedding_cache[text].tolist(),
            "timestamp": datetime.now().isoformat(),
            "importance": importance
        })

    def recall(self, query: str, top_k: int = 3, min_importance: float = 0.4) -> List[Dict]:
        """Поиск в долговременной памяти"""
        if not self.long_term:
            return []
            
        filtered = [m for m in self.long_term if m["importance"] >= min_importance]
        if not filtered:
            return []
            
        query_embed = self._get_embedding(query)
        embeddings = np.array([m["embedding"] for m in filtered])
        similarities = np.dot(embeddings, query_embed)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [{
            "text": filtered[i]["text"],
            "relevance": float(similarities[i])
        } for i in top_indices]

    def _get_embedding(self, text: str) -> np.ndarray:
        """Получение эмбеддинга текста с кэшированием"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.embedder.encode(text)
        return self.embedding_cache[text]

    def get_context(self, max_length: int = 500) -> str:
        """Получение текущего контекста"""
        context = "\n".join(self.context)
        return self.embedder.tokenizer.decode(
            self.embedder.tokenizer.encode(context, max_length=max_length, truncation=True)
        )

    def save(self, path: Union[str, Path]) -> None:
        """Сохранение памяти в файл"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "context": list(self.context),
                "long_term": self.long_term,
                "knowledge_graph": self.knowledge_graph
            }, f, ensure_ascii=False)

    def load(self, path: Union[str, Path]) -> None:
        """Загрузка памяти из файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.context = deque(data.get("context", []), maxlen=self.context.maxlen)
                self.long_term = data.get("long_term", [])
                self.knowledge_graph = data.get("knowledge_graph", [])
        except FileNotFoundError:
            self.logger.info("No memory file found, starting with empty memory")

    def get_recent_topics(self, top_k: int = 3) -> List[str]:
        """Получение последних тем"""
        return [item['query'] for item in self.knowledge_graph[-top_k:]]

    def get_by_emotion(self, emotion: str) -> List[dict]:
        """Фильтрация по эмоциональной окраске"""
        return [item for item in self.knowledge_graph 
                if emotion in item['metadata']['emotion_distribution']]
