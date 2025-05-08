from typing import List, Dict, Optional
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MemorySystem:
    def __init__(self, memory_file: str = "data/memory.json"):
        self.memory_file = Path(memory_file)
        self.memory = self._load_memory()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self._load_embeddings()  # Теперь храним эмбеддинги в numpy массиве
    
    def add_memory(self, content: str, tags: List[str] = None, importance: float = 0.5) -> None:
        """Добавление информации в память"""
        if tags is None:
            tags = []
            
        memory_entry = {
            'content': content,
            'tags': tags,
            'importance': importance,
            'timestamp': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat()
        }
        
        self.memory.append(memory_entry)
        self._update_embeddings(content)
        self._save_memory()
    
    def retrieve_memory(self, query: str, top_k: int = 3) -> List[Dict]:
        """Поиск похожих записей через косинусную схожесть"""
        if len(self.memory) == 0:
            return []

        query_embedding = self.embedder.encode(query).reshape(1, -1)
        
        # Вычисляем схожесть со всеми сохраненными эмбеддингами
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Получаем индексы top_k наиболее похожих записей
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            entry = self.memory[idx].copy()
            entry['similarity'] = float(similarities[idx])  # преобразуем numpy.float32 в python float
            entry['last_accessed'] = datetime.now().isoformat()
            results.append(entry)
        
        self._save_memory()
        return results
    
    def get_recent_topics(self, top_k: int = 3) -> List[str]:
        """Получение недавних тем (без изменений)"""
        sorted_memory = sorted(
            self.memory,
            key=lambda x: x['last_accessed'],
            reverse=True
        )
        return [entry['content'][:100] for entry in sorted_memory[:top_k]]
    
    def _load_memory(self) -> List[Dict]:
        """Загрузка памяти из файла (без изменений)"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def _save_memory(self) -> None:
        """Сохранение памяти в файл (без изменений)"""
        self.memory_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def _load_embeddings(self) -> np.ndarray:
        """Загрузка или создание эмбеддингов"""
        if not self.memory:
            return np.array([])
        
        texts = [entry['content'] for entry in self.memory]
        return self.embedder.encode(texts)
    
    def _update_embeddings(self, new_content: str) -> None:
        """Обновление эмбеддингов с новым контентом"""
        new_embedding = self.embedder.encode([new_content])
        if len(self.embeddings) == 0:
            self.embeddings = new_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])
