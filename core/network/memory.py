# Управление памятью и контекстом
import json
from collections import deque
import numpy as np

class ContextMemory:
    def __init__(self, max_size=10000):
        self.memory = deque(maxlen=100)  # Краткосрочная память
        self.long_term = {}  # Долгосрочное хранилище
        self.embeddings = np.zeros((0, 768))  # Векторные представления

    def add_context(self, text, embedding=None):
        self.memory.append(text)
        if embedding is not None:
            self.embeddings = np.vstack([self.embeddings, embedding])

    def retrieve_relevant(self, query_embedding, top_k=3):
        if len(self.embeddings) == 0:
            return []
        similarities = np.dot(self.embeddings, query_embedding)
        indices = np.argsort(similarities)[-top_k:]
        return [self.memory[i] for i in indices]

    def save(self, path="data/memory.json"):
        with open(path, 'w') as f:
            json.dump({
                'memory': list(self.memory),
                'long_term': self.long_term
            }, f)

    def load(self, path="data/memory.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                self.memory = deque(data['memory'], maxlen=100)
                self.long_term = data['long_term']
        except FileNotFoundError:
            pass
