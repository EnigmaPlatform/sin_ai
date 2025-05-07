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
        self.memory_embeddings = self._load_embeddings()
    
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
        """Поиск в памяти по релевантности"""
        query_embedding = self.embedder.encode(query)
        similarities = cosine_similarity(
            [query_embedding],
            self.memory_embeddings
        )[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            entry = self.memory[idx]
            entry['similarity'] = float(similarities[idx])
            entry['last_accessed'] = datetime.now().isoformat()
            results.append(entry)
        
        self._save_memory()
        return results
    
    def get_recent_topics(self, top_k: int = 3) -> List[str]:
        """Получение недавних тем"""
        sorted_memory = sorted(
            self.memory,
            key=lambda x: x['last_accessed'],
            reverse=True
        )
        return [entry['content'][:100] for entry in sorted_memory[:top_k]]
    
    def _load_memory(self) -> List[Dict]:
        """Загрузка памяти из файла"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def _save_memory(self) -> None:
        """Сохранение памяти в файл"""
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
        if len(self.memory_embeddings) == 0:
            self.memory_embeddings = new_embedding
        else:
            self.memory_embeddings = np.vstack([self.memory_embeddings, new_embedding])
