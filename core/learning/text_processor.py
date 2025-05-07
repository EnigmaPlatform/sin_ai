# Обработка текстовых данных
import re
from collections import Counter
import spacy

class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("ru_core_news_sm")
        self.patterns = {
            'qa': re.compile(r"\[Q\]:(.+?)\[A\]:(.+)", re.DOTALL),
            'code': re.compile(r"```python(.+?)```", re.DOTALL)
        }

    def extract_qa_pairs(self, text):
        return self.patterns['qa'].findall(text)

    def extract_code_blocks(self, text):
        return self.patterns['code'].findall(text)

    def analyze_text(self, text):
        doc = self.nlp(text)
        return {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'topics': self._detect_topics(doc),
            'complexity': len(list(doc.sents)) / max(1, len(doc))
        }

    def _detect_topics(self, doc):
        nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
        return Counter(nouns).most_common(3)
