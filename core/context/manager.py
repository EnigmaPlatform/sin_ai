from collections import deque
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

class ContextManager:
    def __init__(self, max_short_term=10, max_long_term=1000):
        self.short_term_memory = deque(maxlen=max_short_term)
        self.long_term_graph = nx.Graph()
        self.vectorizer = TfidfVectorizer()
        self.entity_cache = {}
    
    def update_context(self, prompt: str, response: str):
        """Обновление контекста с семантическим анализом"""
        # Краткосрочная память
        self.short_term_memory.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now()
        })
        
        # Долгосрочное хранение
        entities = self._extract_entities(prompt)
        self._update_knowledge_graph(entities, response)
    
    def get_relevant_context(self, query: str, top_n: int = 3) -> list:
        """Поиск релевантного контекста"""
        query_embed = self.vectorizer.transform([query])
        scores = []
        
        for node in self.long_term_graph.nodes(data=True):
            node_embed = self.vectorizer.transform([node[1]["text"]])
            similarity = (query_embed * node_embed.T).toarray()[0][0]
            scores.append((node[0], similarity))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    def _extract_entities(self, text: str) -> dict:
        """Извлечение именованных сущностей с кэшированием"""
        if text in self.entity_cache:
            return self.entity_cache[text]
        
        # Реальная реализация использует spaCy или аналоги
        entities = {
            "nouns": re.findall(r"\b[A-Z][a-z]+\b", text),
            "actions": re.findall(r"\b\w+ing\b", text)
        }
        self.entity_cache[text] = entities
        return entities
