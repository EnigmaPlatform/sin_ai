import os
import torch
import json
import requests
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
# Попытка импортировать newspaper, обработка ошибок
try:
    import newspaper
    NEWSPAPER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Библиотека 'newspaper' недоступна: {e}")
    NEWSPAPER_AVAILABLE = False
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tokenizers import ByteLevelBPETokenizer
import numpy as np
from tqdm import tqdm
import pickle
import re
from dataclasses import dataclass
from collections import defaultdict
import Levenshtein  # для нечёткого сравнения имён
import random
import time
# =============== НОВОЕ: Внешние библиотеки ===============
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
# =============================================
# 🔹 ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# =============================================
ROOT_DIR = r"C:\Users\User\Downloads\Sin"
MODEL_DIR = os.path.join(ROOT_DIR, "model")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "tokenizer")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHAT_DIR = os.path.join(ROOT_DIR, "chat_memory")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MEMORY_DIR = os.path.join(ROOT_DIR, "memory")
GRAPH_DIR = os.path.join(ROOT_DIR, "graph")
CONFIG_FILE = os.path.join(ROOT_DIR, "sin_config.json")
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "raw", "scraped"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "raw", "uploaded"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
# Настройка логирования
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
# =============================================
# 🔹 КОНФИГУРАЦИЯ
# =============================================
DEFAULT_CONFIG = {
    "model_name": "distilgpt2",
    "current_version": "v2.1",
    "learning_progress": 0.0,
    "total_tokens_seen": 0,
    "unique_concepts": 0,
    "last_trained": None,
    "tokenizer_updates": 0,
    "memory_entries": 0,
    "graph_nodes": 0,
    "unsupervised_generated": 0,
    "unsupervised_learned": 0,
}
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Ошибка чтения config.json: {e}. Используются настройки по умолчанию.")
    return DEFAULT_CONFIG.copy()

def save_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Ошибка сохранения config.json: {e}")

config = load_config()
# =============================================
# 🔹 ТОКЕНИЗАТОР С АДАПТАЦИЕЙ
# =============================================
class AdaptiveTokenizer:
    def __init__(self):
        self.tokenizer = None
        self.vocab_file = os.path.join(TOKENIZER_DIR, "vocab.json")
        self.merges_file = os.path.join(TOKENIZER_DIR, "merges.txt")
        self.load_or_init()
    def load_or_init(self):
        if os.path.exists(self.vocab_file) and os.path.exists(self.merges_file):
            try:
                self.tokenizer = GPT2TokenizerFast(
                    vocab_file=self.vocab_file,
                    merges_file=self.merges_file
                )
                logging.info("Токенизатор загружен.")
            except Exception as e:
                logging.error(f"Ошибка загрузки токенизатора из файлов: {e}. Инициализация с distilgpt2.")
                self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
                self.save()
        else:
            self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            self.save()
            logging.info("Токенизатор инициализирован с DistilGPT-2.")
        # Убедимся, что есть pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logging.info("Добавлен pad_token.")

    def save(self):
        try:
            self.tokenizer.save_pretrained(TOKENIZER_DIR)
            logging.info("Токенизатор сохранён.")
        except Exception as e:
             logging.error(f"Ошибка сохранения токенизатора: {e}")

    def update_from_corpus(self, texts):
        special_tokens = ["[USER]", "[SIN]", "[URL]", "[FILE]", "[MEM]", "[GRAPH]", "[PAD]"]
        new_tokens = []
        for text in texts:
            # Более надежный способ извлечения слов: учитываем апострофы, дефисы
            words = re.findall(r'\b[A-Z][a-z\']*(?:[-][A-Z][a-z\']*)*\b', text)
            for word in words:
                if len(word) > 1 and word not in self.tokenizer.vocab:
                    new_tokens.append(word)
        # Ограничиваем количество новых токенов за раз
        all_new = special_tokens + list(set(new_tokens))[:20]
        if all_new:
            self.tokenizer.add_tokens(all_new)
            # Обновляем pad_token_id если он был добавлен
            if '[PAD]' in all_new:
                 self.tokenizer.pad_token = '[PAD]'
            config["tokenizer_updates"] += 1
            save_config(config)
            logging.info(f"Токенизатор обновлён: добавлено {len(all_new)} токенов.")
        return len(all_new)
# =============================================
# 🔹 ВЕКТОРНАЯ ПАМЯТЬ (RAG)
# =============================================
class VectorMemory:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_path = os.path.join(MEMORY_DIR, "vector.index")
        self.kb_path = os.path.join(MEMORY_DIR, "knowledge.pkl")
        self.index = None
        self.knowledge_base = []  # list of {"text": str, "source": str, "timestamp": iso}
        self.load()
    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.kb_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.kb_path, "rb") as f:
                    self.knowledge_base = pickle.load(f)
                logging.info(f"Векторная память загружена: {len(self.knowledge_base)} записей.")
            except Exception as e:
                 logging.error(f"Ошибка загрузки векторной памяти: {e}. Инициализация новой.")
                 self._init_new_memory()
        else:
            self._init_new_memory()
            
    def _init_new_memory(self):
        d = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(d)  # простой индекс
        logging.info("Векторная память инициализирована.")

    def save(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.kb_path, "wb") as f:
                pickle.dump(self.knowledge_base, f)
            config["memory_entries"] = len(self.knowledge_base)
            save_config(config)
        except Exception as e:
             logging.error(f"Ошибка сохранения векторной памяти: {e}")

    def add(self, text, source="unknown"):
        if len(text.strip()) < 5:
            return
        try:
            embedding = self.encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            self.index.add(embedding)
            self.knowledge_base.append({
                "text": text,
                "source": source,
                "timestamp": datetime.now().isoformat()
            })
            logging.info(f"Память: добавлено знание из {source}")
        except Exception as e:
            logging.error(f"Ошибка добавления в векторную память: {e}")

    def retrieve(self, query, k=3):
        if self.index.ntotal == 0:
            return []
        try:
            query_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            D, I = self.index.search(query_emb, min(k, self.index.ntotal)) # Предотвращаем ошибку, если k > ntotal
            results = [self.knowledge_base[i]["text"] for i in I[0] if i < len(self.knowledge_base)]
            return results
        except Exception as e:
             logging.error(f"Ошибка поиска в векторной памяти: {e}")
             return []

# =============================================
# 🔹 ГРАФ ЗНАНИЙ
# =============================================
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        # Улучшенные паттерны
        self.patterns = [
            (r'([A-Za-zА-Яа-яЁё\s\-\'\"]+?)\s+—\s+это\s+([A-Za-zА-Яа-яЁё\s\-\'\"]+)', 'is_a'),
            (r'([A-Za-zА-Яа-яЁё\s\-\'\"]+?)\s+является\s+([A-Za-zА-Яа-яЁё\s\-\'\"]+)', 'is_a'),
            (r'([A-Za-zА-Яа-яЁё\s\-\'\"]+?)\s+делает\s+([A-Za-zА-Яа-яЁё\s\-\'\"]+)', 'does'),
            (r'([A-Za-zА-Яа-яЁё\s\-\'\"]+?)\s+любит\s+([A-Za-zА-Яа-яЁё\s\-\'\"]+)', 'loves'),
            (r'([A-Za-zА-Яа-яЁё\s\-\'\"]+?)\s+находится\s+в\s+([A-Za-zА-Яа-яЁё\s\-\'\"]+)', 'located_in'),
            (r'([A-Za-zА-Яа-яЁё\s\-\'\"]+?)\s+часть\s+([A-Za-zА-Яа-яЁё\s\-\'\"]+)', 'part_of'),
            (r'([A-Za-zА-Яа-яЁё\s\-\'\"]+?)\s+имеет\s+([A-Za-zА-Яа-яЁё\s\-\'\"]+)', 'has'),
        ]
        self.save_path = os.path.join(GRAPH_DIR, "graph.gml")
        self.load()
    def load(self):
        if os.path.exists(self.save_path):
            try:
                self.graph = nx.read_gml(self.save_path)
                logging.info(f"Граф знаний загружен: {self.graph.number_of_nodes()} узлов.")
            except Exception as e:
                 logging.error(f"Ошибка загрузки графа: {e}")
                 # Инициализируем новый граф, если загрузка не удалась
                 self.graph = nx.DiGraph()
                 logging.info("Граф знаний инициализирован (ошибка загрузки).")
        else:
            logging.info("Граф знаний инициализирован.")
    def save(self):
        try:
            nx.write_gml(self.graph, self.save_path)
            config["graph_nodes"] = self.graph.number_of_nodes()
            save_config(config)
        except Exception as e:
             logging.error(f"Ошибка сохранения графа: {e}")

    def extract_triples(self, text):
        triples = []
        # Разбиваем на предложения более надежно
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent: continue
            # Пробуем паттерны
            for pattern, rel in self.patterns:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match:
                    subj, obj = match.groups()
                    # Очищаем от лишних пробелов
                    subj_clean = subj.strip()
                    obj_clean = obj.strip()
                    if subj_clean and obj_clean:
                        triples.append((subj_clean, rel, obj_clean))
        return triples

    def add_knowledge(self, text):
        triples = self.extract_triples(text)
        added_count = 0
        for subj, rel, obj in triples:
            # Добавляем узлы, если их нет
            if not self.graph.has_node(subj):
                self.graph.add_node(subj)
            if not self.graph.has_node(obj):
                self.graph.add_node(obj)
            # Добавляем ребро (если оно уже существует, nx не добавит дубликат)
            self.graph.add_edge(subj, obj, relation=rel)
            added_count += 1
        if added_count:
            logging.info(f"Граф: добавлено/обновлено {added_count} связей.")

    def get_context(self, query, depth=2):
        if self.graph.number_of_nodes() == 0:
            return []
        query = query.lower()
        # Поиск ближайших узлов
        close_nodes = [n for n in self.graph.nodes if Levenshtein.distance(query, n.lower()) < 3]
        nodes = [query] + close_nodes
        context = []
        for node in nodes:
            # Проверяем, существует ли узел в графе
            if node in self.graph:
                # Исходящие связи
                for neighbor in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if edge_data and 'relation' in edge_data:
                        relation = edge_data['relation']
                        context.append(f"{node} {relation} {neighbor}")
                # Входящие связи
                for predecessor in self.graph.predecessors(node):
                    edge_data = self.graph.get_edge_data(predecessor, node)
                    if edge_data and 'relation' in edge_data:
                         relation = edge_data['relation']
                         context.append(f"{predecessor} {relation} {node}")

        return context[:5] # Ограничиваем количество

# =============================================
# 🔹 МОДЕЛЬ С НЕПРЕРЫВНЫМ ОБУЧЕНИЕМ
# =============================================
class SinModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_or_init()
    def load_or_init(self):
        try:
            # Попробуем загрузить модель и токенизатор из MODEL_DIR
            if os.path.exists(os.path.join(MODEL_DIR, "config.json")): # Проверяем наличие модели
                self.model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
                self.tokenizer = AdaptiveTokenizer().tokenizer # Перезагружаем токенизатор из файла
                self.model.resize_token_embeddings(len(self.tokenizer))
                logging.info("Модель Sin загружена из директории.")
            else:
                # Инициализируем новую модель
                self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
                self.tokenizer = AdaptiveTokenizer().tokenizer
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.save() # Сохраняем инициализированную модель
                logging.info("Новая модель Sin инициализирована.")
        except Exception as e:
            logging.error(f"Ошибка загрузки/инициализации модели: {e}")
            # В крайнем случае, создаем новую модель (может не совпадать с сохраненной)
            self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self.tokenizer = AdaptiveTokenizer().tokenizer
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.warning("Резервная инициализация модели.")

    def save(self):
        try:
            self.model.save_pretrained(MODEL_DIR)
            # Токенизатор сохраняется отдельно AdaptiveTokenizer
            logging.info("Модель Sin сохранена.")
        except Exception as e:
             logging.error(f"Ошибка сохранения модели: {e}")


    def fine_tune(self, texts, epochs=1):
        if not texts or all(len(t.strip()) <= 10 for t in texts):
             logging.info("Нет данных для дообучения или тексты слишком короткие.")
             return

        from datasets import Dataset
        def preprocess(text):
            return " ".join(text.strip().split()[:512]) # strip перед split
        cleaned_texts = [preprocess(t) for t in texts if len(t.strip()) > 10]
        if len(cleaned_texts) == 0:
            logging.info("Нет подходящих текстов для дообучения после очистки.")
            return
        try:
            dataset = Dataset.from_dict({"text": cleaned_texts})
            def tokenize_function(examples):
                # Убедимся, что токенизатор доступен
                if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                    logging.error("Токенизатор не инициализирован для дообучения.")
                    return None
                return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128) # Используем padding="max_length"
            tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], batch_size=4) # Добавим размер батча

            # Фильтруем None результаты токенизации (на случай ошибок)
            tokenized_datasets = tokenized_datasets.filter(lambda example: example is not None and all(k in example for k in ['input_ids', 'attention_mask']))

            if len(tokenized_datasets) == 0:
                logging.info("Нет данных после токенизации для дообучения.")
                return

            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            # Исправление: num_train_epochs должно быть int. Симуляция дробной эпохи через max_steps.
            effective_epochs = max(1, int(epochs)) # Принимаем минимум 1 эпоху
            total_steps = int((len(tokenized_datasets) / 4) * effective_epochs) # batch_size=4
            # max_steps_for_fractional = max(1, int(total_steps * (epochs % 1))) if epochs != int(epochs) else total_steps

            training_args = TrainingArguments(
                output_dir=MODEL_DIR,
                overwrite_output_dir=True,
                num_train_epochs=effective_epochs, # Используем целое число
                # max_steps=max_steps_for_fractional if epochs != int(epochs) else total_steps, # Раскомментируйте, если хотите точную дробь, но это сложно
                per_device_train_batch_size=2, # Уменьшаем batch_size для стабильности
                gradient_accumulation_steps=2, # Компенсируем уменьшение batch_size
                save_steps=10_000,
                save_total_limit=1,
                logging_dir=LOGS_DIR,
                logging_steps=100,
                report_to=[],
                no_cuda=not torch.cuda.is_available(),
                # disable_tqdm=False # Опционально, для отображения прогресса
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_datasets,
            )
            trainer.train() # Убираем resume_from_checkpoint, так как мы перезаписываем
            self.save()
            config["learning_progress"] = min(100.0, config["learning_progress"] + len(texts) * 0.15)
            config["total_tokens_seen"] += sum(len(self.tokenizer.encode(t)) for t in cleaned_texts)
            # Исправление: правильно считаем уникальные слова
            all_words = [word for t in cleaned_texts for word in t.split()]
            config["unique_concepts"] = len(set(all_words))
            config["last_trained"] = datetime.now().isoformat()
            save_config(config)
            logging.info(f"Дообучение завершено: {len(cleaned_texts)} текстов.")
        except Exception as e:
             logging.error(f"Ошибка в процессе дообучения: {e}")

# =============================================
# 🔹 ПАРСИНГ И ОЧИСТКА
# =============================================
def scrape_url(url):
    text = ""
    if NEWSPAPER_AVAILABLE:
        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()
            if article.text and len(article.text) > 100:
                return article.text
            else:
                 logging.info(f"Newspaper не нашел достаточного текста для {url}")
        except Exception as e:
            logging.warning(f"Ошибка парсинга URL {url} с newspaper: {e}. Пробуем requests+BeautifulSoup.")

    # Резервный метод парсинга
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15) # Увеличен timeout
        response.raise_for_status() # Проверка статуса ответа
        soup = BeautifulSoup(response.content, "html.parser")
        # Более агрессивная очистка
        for elem in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            elem.decompose()
        # Попробуем найти основной контент (часто в <main>, <article>)
        main_content = soup.find('main') or soup.find('article')
        if main_content:
            text = " ".join(main_content.stripped_strings)
        else:
            text = " ".join(soup.stripped_strings)

        # Ограничиваем длину текста
        text = text[:15000] # Увеличен лимит
        if len(text) > 100:
             logging.info(f"Текст извлечен с requests+BeautifulSoup для {url}, длина: {len(text)}")
             return text
        else:
             logging.warning(f"Текст, извлеченный с requests+BeautifulSoup для {url}, слишком короткий.")
             return ""
    except requests.exceptions.RequestException as e:
         logging.error(f"Ошибка запроса к URL {url}: {e}")
         return ""
    except Exception as e:
        logging.error(f"Ошибка парсинга URL {url} с requests+BeautifulSoup: {e}")
        return ""

# =============================================
# 🔹 ЧТЕНИЕ ФАЙЛОВ
# =============================================
def read_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".docx":
            # Попытка импорта docx
            try:
                import docx
                doc = docx.Document(filepath)
                return "\n".join(paragraph.text for paragraph in doc.paragraphs)
            except ImportError:
                print("Библиотека 'python-docx' не установлена. Установите её для чтения .docx файлов.")
                logging.error("Библиотека 'python-docx' не установлена.")
                return ""
        elif ext == ".pdf":
            # Попытка импорта PyPDF2
            try:
                import PyPDF2
                text = ""
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                 print("Библиотека 'PyPDF2' не установлена. Установите её для чтения .pdf файлов.")
                 logging.error("Библиотека 'PyPDF2' не установлена.")
                 return ""
            except Exception as e:
                 logging.error(f"Ошибка чтения PDF {filepath}: {e}")
                 return ""
        elif ext == ".json":
            # Попытка чтения JSON
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Простая обработка: если это словарь с ключом 'text', берем его.
                # Если это массив, объединяем все элементы в строку.
                if isinstance(data, dict) and 'text' in data:
                     return str(data['text'])
                elif isinstance(data, list):
                     return " ".join(str(item) for item in data)
                elif isinstance(data, str):
                     return data # Если JSON содержит просто строку
                else:
                     # Попробуем преобразовать весь словарь в строку
                     return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError as e:
                 logging.error(f"Ошибка парсинга JSON {filepath}: {e}")
                 print(f"Ошибка парсинга JSON файла {filepath}: {e}")
                 return ""
            except Exception as e:
                 logging.error(f"Неизвестная ошибка при чтении JSON {filepath}: {e}")
                 print(f"Неизвестная ошибка при чтении JSON файла {filepath}: {e}")
                 return ""
        else:
            print(f"Неподдерживаемый формат файла: {ext}")
            return ""
    except Exception as e:
        logging.error(f"Ошибка чтения файла {filepath}: {e}")
        return ""
# =============================================
# 🔹 ОБУЧЕНИЕ ИЗ ИСТОЧНИКОВ
# =============================================
def learn_from_url(url):
    print("Начинаю извлекать информацию с URL...")
    text = scrape_url(url)
    if len(text) < 50:
        print("❌ Не удалось извлечь достаточный текст с URL.")
        return
    save_path = os.path.join(DATA_DIR, "raw", "scraped", f"{abs(hash(url))}.txt") # Используем abs для положительного имени файла
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Текст сохранен локально.")
    except Exception as e:
         logging.error(f"Ошибка сохранения текста с URL: {e}")
         print("Ошибка при сохранении текста локально.")

    # Передаем текст в систему
    sin.fine_tune([text])
    memory.add(text, source=f"url:{url}")
    kg.add_knowledge(text)
    # Сохраняем состояние системы
    try:
        memory.save()
        kg.save()
    except Exception as e:
         logging.error(f"Ошибка сохранения памяти/графа после обучения с URL: {e}")
    print("✅ Обучение с URL завершено.")

def learn_from_file(filepath):
    print("Начинаю читать файл...")
    text = read_file(filepath)
    if len(text) < 50:
        print("❌ Не удалось прочитать файл или он пуст/слишком короткий.")
        return
    filename = os.path.basename(filepath)
    save_path = os.path.join(DATA_DIR, "raw", "uploaded", filename + ".txt")
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Текст из файла сохранен локально.")
    except Exception as e:
         logging.error(f"Ошибка сохранения текста из файла: {e}")
         print("Ошибка при сохранении текста из файла локально.")

    # Передаем текст в систему
    sin.fine_tune([text])
    memory.add(text, source=f"file:{filename}")
    kg.add_knowledge(text)
    # Сохраняем состояние системы
    try:
        memory.save()
        kg.save()
    except Exception as e:
         logging.error(f"Ошибка сохранения памяти/графа после обучения с файла: {e}")
    print("✅ Обучение с файла завершено.")

# =============================================
# 🔹 ОБУЧЕНИЕ БЕЗ ВНЕШНЕЙ ИНФОРМАЦИИ (Unsupervised)
# =============================================
def learn_unsupervised(hours=1):
    print(f"🧠 Начинаю обучение без внешней информации на {hours} час(ов)...")
    print("Это может занять некоторое время. Статистика будет обновляться.")
    start_time = time.time()
    end_time = start_time + hours * 3600
    generated_count = 0
    learned_count = 0
    
    # Получаем текущий словарь для генерации
    vocab_words = list(sin.tokenizer.get_vocab().keys())
    # Фильтруем специальные токены
    vocab_words = [w for w in vocab_words if not w.startswith('[') and not w.startswith('<')]

    log_path = os.path.join(LOGS_DIR, "unsupervised_learning.log")
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n--- Начало обучения без информации {datetime.now().isoformat()} ---\n")

    try:
        while time.time() < end_time:
            # 1. Генерация: используем случайные слова из словаря как затравку
            seed_words = random.sample(vocab_words, k=min(5, len(vocab_words)))
            seed_text = " ".join(seed_words)
            
            # prompt = f"[SIN] {seed_text}" # Простой промпт
            prompt = seed_text # Простой промпт
            
            try:
                inputs = sin.tokenizer.encode(prompt, return_tensors="pt", max_length=50, truncation=True)
                
                # Генерируем короткий текст
                outputs = sin.model.generate(
                    inputs,
                    max_length=min(inputs.shape[1] + 50, 200), # Ограничиваем длину
                    num_return_sequences=1,
                    temperature=0.9,
                    top_k=40,
                    pad_token_id=sin.tokenizer.eos_token_id,
                    do_sample=True,
                    # no_repeat_ngram_size=2, # Может быть слишком строгим для коротких текстов
                )
                generated_text = sin.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Убираем затравку
                if generated_text.startswith(prompt):
                    response_text = generated_text[len(prompt):].strip()
                else:
                    response_text = generated_text.strip()
                
                generated_count += 1
                
                # 2. Фильтрация: очень простая - проверяем длину и наличие слов
                if len(response_text) > 10 and len(response_text.split()) > 2:
                    # Можно добавить более сложные фильтры (проверка на повторы, грамматику и т.д.)
                    # Пока просто добавляем
                    # print(f"Сгенерировано: {response_text[:100]}...")
                    
                    # 3. Добавление в систему
                    sin.fine_tune([response_text], epochs=0.1) # Очень короткое дообучение
                    memory.add(response_text, source="unsupervised")
                    kg.add_knowledge(response_text)
                    learned_count += 1
                    
                    with open(log_path, "a", encoding="utf-8") as log_f:
                        log_f.write(f"[{datetime.now().isoformat()}] Осмысленный: {response_text}\n")
                
                # Обновляем статистику в конфиге
                config["unsupervised_generated"] = generated_count
                config["unsupervised_learned"] = learned_count
                save_config(config)
                
                # Периодическое сохранение состояния
                if generated_count % 20 == 0: # Каждые 20 генераций
                    try:
                        memory.save()
                        kg.save()
                        sin.save() # Сохраняем модель реже, например, каждые 100
                        if generated_count % 100 == 0:
                            print(f"  🔄 Промежуточное сохранение. Сгенерировано: {generated_count}, Освоено: {learned_count}")
                    except Exception as e:
                         logging.error(f"Ошибка промежуточного сохранения: {e}")
                
                # Небольшая пауза, чтобы не перегружать CPU
                time.sleep(0.5) 
                
            except Exception as gen_e:
                 logging.error(f"Ошибка генерации/дообучения в unsupervised: {gen_e}")
                 # Не останавливаем весь процесс из-за одной ошибки
                 continue

    except KeyboardInterrupt:
         print("\n⚠️  Обучение без информации прервано пользователем.")
    finally:
        elapsed_time = time.time() - start_time
        print(f"✅ Обучение без информации завершено за {elapsed_time/3600:.2f} часов.")
        print(f"📊 Сгенерировано предложений: {generated_count}")
        print(f"📊 Освоено предложений: {learned_count}")
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"--- Завершено {datetime.now().isoformat()}. Сгенерировано: {generated_count}, Освоено: {learned_count} ---\n")
        
        # Финальное сохранение
        try:
            memory.save()
            kg.save()
            sin.save()
        except Exception as e:
             logging.error(f"Ошибка финального сохранения в unsupervised: {e}")

# =============================================
# 🔹 ЧАТ С RAG + ГРАФОМ
# =============================================
def chat():
    print("\n" + "="*50)
    print("💬 Чат с Sin. Введите 'выход', 'обучение', 'url', 'файл', 'обучение_без_инфо' для управления.")
    print("="*50)
    dialogue_history = []
    # Выносим log_path за цикл
    log_path = os.path.join(CHAT_DIR, "dialogues.log")
    while True:
        user_input = input("\n[Вы]: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "выход":
            break
        elif user_input.lower() == "обучение":
            show_learning_progress()
            continue
        elif user_input.lower() == "url":
            url = input("Введите URL: ").strip()
            if url:
                learn_from_url(url)
            else:
                print("URL не введен.")
            continue
        elif user_input.lower() == "файл":
            path = input("Путь к файлу (.txt, .docx, .pdf, .json): ").strip()
            if os.path.exists(path):
                learn_from_file(path)
            else:
                print("❌ Файл не найден.")
            continue
        elif user_input.lower() == "обучение_без_инфо":
            try:
                hours = float(input("Введите продолжительность обучения в часах (например, 0.5 для 30 минут): "))
                if hours > 0:
                    learn_unsupervised(hours)
                else:
                    print("Количество часов должно быть положительным.")
            except ValueError:
                print("Введите корректное число.")
            continue

        # RAG: поиск знаний
        try:
            retrieved = memory.retrieve(user_input, k=2)
        except Exception as e:
             logging.error(f"Ошибка при извлечении из памяти: {e}")
             retrieved = []
        try:
            graph_context = kg.get_context(user_input)
        except Exception as e:
             logging.error(f"Ошибка при извлечении из графа: {e}")
             graph_context = []

        context = ""
        if retrieved:
            context += "[MEM] " + " ".join(retrieved[:2]) + " "
        if graph_context:
            context += "[GRAPH] " + " | ".join(graph_context) + " "

        # Генерация
        prompt = f"{context}[USER] {user_input} [SIN]"
        try:
            inputs = sin.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True) # Добавим truncation
            outputs = sin.model.generate(
                inputs,
                max_length=min(inputs.shape[1] + 150, 512), # Увеличил длину генерации
                num_return_sequences=1,
                temperature=0.8, # Немного понизили для стабильности
                top_k=50,
                pad_token_id=sin.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=True,
                # early_stopping=True # Опционально
            )
            response = sin.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Более надежное удаление промпта из ответа
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            else:
                # Если точное совпадение не найдено, пробуем найти и удалить [USER] часть
                user_marker_idx = response.find("[USER]")
                sin_marker_idx = response.find("[SIN]", user_marker_idx)
                if sin_marker_idx != -1:
                    response = response[sin_marker_idx + len("[SIN]"):].strip()
                else:
                    # В крайнем случае, просто удаляем начало, если оно совпадает частично
                    if response.startswith("[SIN]"):
                        response = response[len("[SIN]"):].strip()

            if not response or len(response) < 2: # Проверка на пустоту
                response = "Я пока не знаю, как ответить на это, но запомню."

        except Exception as e:
             logging.error(f"Ошибка генерации ответа: {e}", exc_info=True) # Добавляем трассировку
             response = "Произошла ошибка при генерации ответа. Попробуйте еще раз."

        print(f"[Sin]: {response}")

        # Сохраняем диалог
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} | {user_input} || {response}\n")
        except Exception as e:
             logging.error(f"Ошибка записи диалога в файл: {e}")

         
     print("Выполняется краткое дообучение на последнем диалоге...") # Информирование пользователя
         sin.fine_tune([f"[USER] {user_input}", f"[SIN] {response}"], epochs=1) # Используем 1 эпоху
         memory.add(user_input, source="chat")
         memory.add(response, source="chat")
         kg.add_knowledge(user_input)
         kg.add_knowledge(response)
         try:
             memory.save()
             kg.save()
         except Exception as e:
              logging.error(f"Ошибка сохранения после диалога: {e}")
    # Финальное сохранение модели при выходе из чата
    try:
        sin.save()
    except Exception as e:
         logging.error(f"Ошибка финального сохранения модели: {e}")

def show_learning_progress():
    progress = config.get("learning_progress", 0)
    tokens = config.get("total_tokens_seen", 0)
    concepts = config.get("unique_concepts", 0)
    mem = config.get("memory_entries", 0)
    nodes = config.get("graph_nodes", 0)
    unsup_gen = config.get("unsupervised_generated", 0)
    unsup_learn = config.get("unsupervised_learned", 0)
    bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
    color = "🟢" if progress > 70 else "🟡" if progress > 40 else "🔴"
    print("\n📊 **Статус обучения Sin**")
    print(f"{color} Прогресс: {progress:.1f}% [{bar}]")
    print(f"📚 Уникальных понятий: {concepts}")
    print(f"🔖 Всего токенов: {tokens:,}")
    print(f"🧠 Записей в памяти: {mem}")
    print(f"🕸️  Узлов в графе: {nodes}")
    print(f"🤖 Самообучение (сгенерировано/освоено): {unsup_gen}/{unsup_learn}")
    last_trained = config.get("last_trained")
    if last_trained:
        try:
            # Попробуем распарсить дату
            dt = datetime.fromisoformat(last_trained.replace('Z', '+00:00')) # Обработка 'Z'
            print(f"🕐 Последнее обучение: {dt.strftime('%Y-%m-%d %H:%M')}")
        except ValueError:
            print(f"🕐 Последнее обучение: {last_trained[:16] if last_trained else 'Никогда'}")

# =============================================
# 🔹 ОСНОВНОЕ МЕНЮ
# =============================================
def main():
    global sin, memory, kg
    print("🧠 Загрузка Sin — имитации когнитивной модели...")
    try:
        sin = SinModel()
        memory = VectorMemory()
        kg = KnowledgeGraph()
        tokenizer = AdaptiveTokenizer() # Инициализируем для создания/загрузки
        print(f"✅ Sin готова. Уровень обучения: {config.get('learning_progress', 0):.1f}%")
        print("Введите 'помощь' для списка команд.")
    except Exception as e:
         logging.critical(f"Критическая ошибка при инициализации: {e}")
         print(f"Критическая ошибка при инициализации Sin: {e}")
         return # Завершаем программу, если инициализация не удалась

    while True:
        cmd = input("\n> ").strip().lower()
        if cmd in ["помощь", "help"]:
            print("""
Доступные команды:
- чат                 — начать диалог с Sin
- обучение            — показать прогресс обучения
- url                 — обучить по ссылке
- файл                — обучить по файлу
- обучение_без_инфо   — обучение без внешней информации
- выход               — выйти
            """)
        elif cmd == "чат":
            chat()
        elif cmd == "обучение":
            show_learning_progress()
        elif cmd == "url":
            url = input("URL: ").strip()
            if url:
                learn_from_url(url)
            else:
                 print("URL не введен.")
        elif cmd == "файл":
            path = input("Путь к файлу: ").strip()
            if os.path.exists(path):
                learn_from_file(path)
            else:
                print("❌ Файл не найден.")
        elif cmd == "обучение_без_инфо":
            try:
                hours = float(input("Введите продолжительность обучения в часах (например, 0.5 для 30 минут): "))
                if hours > 0:
                    learn_unsupervised(hours)
                else:
                    print("Количество часов должно быть положительным.")
            except ValueError:
                print("Введите корректное число.")
        elif cmd == "выход":
            # Финальное сохранение всех компонентов
            try:
                sin.save()
                memory.save()
                kg.save()
                save_config(config)
                print("👋 Sin сохранена. До встречи.")
            except Exception as e:
                 logging.error(f"Ошибка при финальном сохранении: {e}")
                 print("Ошибка при финальном сохранении.")
            break
        else:
            print("Неизвестная команда. Введите 'помощь'.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
        # Пытаемся сохранить состояние при прерывании
        try:
            if 'sin' in globals(): sin.save()
            if 'memory' in globals(): memory.save()
            if 'kg' in globals(): kg.save()
            if 'config' in globals(): save_config(config)
        except: pass # Игнорируем ошибки сохранения при прерывании
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}", exc_info=True) # Добавляем трассировку стека
        print(f"Критическая ошибка: {e}")
