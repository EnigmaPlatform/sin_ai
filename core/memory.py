from typing import List, Dict, Optional
from annoy import AnnoyIndex 
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
        self.index = AnnoyIndex(self.embedder.get_sentence_embedding_dimension(), 'angular')
        self._build_index()
    
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
    
    def _build_index(self):
        """Построение индекса для быстрого поиска"""
        self.index.unload()
        if len(self.memory) == 0:
            return
            
        embeddings = self.embedder.encode([m['content'] for m in self.memory])
        for i, emb in enumerate(embeddings):
            self.index.add_item(i, emb)
        self.index.build(10)  # 10 деревьев
    
    def retrieve_memory(self, query: str, top_k: int = 3) -> List[Dict]:
        """Поиск с использованием векторного индекса"""
        query_embedding = self.embedder.encode(query)
        indices = self.index.get_nns_by_vector(
            query_embedding, 
            top_k,
            include_distances=True
        )
        
        results = []
        for idx, distance in zip(*indices):
            entry = self.memory[idx]
            entry['similarity'] = 1 - distance  # преобразуем расстояние в схожесть
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
