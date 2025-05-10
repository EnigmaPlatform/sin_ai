import json
import numpy as np
from collections import deque
from datetime import datetime
from sentence_transformers import SentenceTransformer

class SinMemory:
    def __init__(self, max_context=5):
        self.context = deque(maxlen=max_context)
        self.long_term = []
        self.embedder = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        
    def add_interaction(self, user_text, ai_text):
        self.context.append(f"User: {user_text}")
        self.context.append(f"Sin: {ai_text}")
        self._evaluate_importance(user_text)
    
    def _evaluate_importance(self, text):
        # Эвристики для определения важности
        importance = 0.3  # Базовая важность
        
        if len(text.split()) > 7:
            importance += 0.2
        if '?' in text:
            importance += 0.1
        if any(word in text.lower() for word in ['важно', 'запомни']):
            importance += 0.4
            
        if importance > 0.5:
            self.remember(text, importance)
    
    def remember(self, text, importance=0.5):
        embedding = self.embedder.encode(text)
        self.long_term.append({
            "text": text,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now().isoformat(),
            "importance": importance
        })
    
    def recall(self, query, top_k=3, min_importance=0.4):
        if not self.long_term:
            return []
            
        query_embed = self.embedder.encode(query)
        embeddings = np.array([m["embedding"] for m in self.long_term 
                             if m["importance"] >= min_importance])
        
        if len(embeddings) == 0:
            return []
            
        similarities = np.dot(embeddings, query_embed)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [{
            "text": self.long_term[i]["text"],
            "relevance": float(similarities[i])
        } for i in top_indices]
    
    def get_context(self):
        return "\n".join(self.context)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "context": list(self.context),
                "long_term": self.long_term
            }, f, ensure_ascii=False)
    
    def load(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.context = deque(data["context"], maxlen=self.context.maxlen)
                self.long_term = data.get("long_term", [])
        except FileNotFoundError:
            pass
