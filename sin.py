
import os
import torch
import json
import requests
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import newspaper
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tokenizers import ByteLevelBPETokenizer
import numpy as np
from tqdm import tqdm
import pickle
import re
from dataclasses import dataclass
from collections import defaultdict
import Levenshtein  # для нечёткого сравнения имён

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
    "current_version": "v2.0",
    "learning_progress": 0.0,
    "total_tokens_seen": 0,
    "unique_concepts": 0,
    "last_trained": None,
    "tokenizer_updates": 0,
    "memory_entries": 0,
    "graph_nodes": 0
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

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
            self.tokenizer = GPT2TokenizerFast(
                vocab_file=self.vocab_file,
                merges_file=self.merges_file
            )
            logging.info("Токенизатор загружен.")
        else:
            self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            self.save()
            logging.info("Токенизатор инициализирован с DistilGPT-2.")

    def save(self):
        self.tokenizer.save_pretrained(TOKENIZER_DIR)
        logging.info("Токенизатор сохранён.")

    def update_from_corpus(self, texts):
        special_tokens = ["[USER]", "[SIN]", "[URL]", "[FILE]", "[MEM]", "[GRAPH]"]
        new_tokens = []
        for text in texts:
            words = re.findall(r'\b[A-Z][a-z]+\b', text)  # простые сущности (собственные имена)
            for word in words:
                if len(word) > 2 and word not in self.tokenizer.vocab:
                    new_tokens.append(word)
        all_new = special_tokens + new_tokens[:20]  # максимум 20 новых
        if all_new:
            self.tokenizer.add_tokens(all_new)
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
            self.index = faiss.read_index(self.index_path)
            with open(self.kb_path, "rb") as f:
                self.knowledge_base = pickle.load(f)
            logging.info(f"Векторная память загружена: {len(self.knowledge_base)} записей.")
        else:
            d = self.encoder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(d)  # простой индекс
            logging.info("Векторная память инициализирована.")

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.kb_path, "wb") as f:
            pickle.dump(self.knowledge_base, f)
        config["memory_entries"] = len(self.knowledge_base)
        save_config(config)

    def add(self, text, source="unknown"):
        if len(text.strip()) < 5:
            return
        embedding = self.encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embedding)
        self.knowledge_base.append({
            "text": text,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        logging.info(f"Память: добавлено знание из {source}")

    def retrieve(self, query, k=3):
        if self.index.ntotal == 0:
            return []
        query_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(query_emb, k)
        results = [self.knowledge_base[i]["text"] for i in I[0]]
        return results


# =============================================
# 🔹 ГРАФ ЗНАНИЙ
# =============================================

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.patterns = [
            (r'([A-Za-z]+) — это ([A-Za-z]+)', 'is_a'),
            (r'([A-Za-z]+) является ([A-Za-z]+)', 'is_a'),
            (r'([A-Za-z]+) делает ([A-Za-z]+)', 'does'),
            (r'([A-Za-z]+) любит ([A-Za-z]+)', 'loves'),
            (r'([A-Za-z]+) находится в ([A-Za-z]+)', 'located_in'),
        ]
        self.save_path = os.path.join(GRAPH_DIR, "graph.gml")
        self.load()

    def load(self):
        if os.path.exists(self.save_path):
            self.graph = nx.read_gml(self.save_path)
            logging.info(f"Граф знаний загружен: {self.graph.number_of_nodes()} узлов.")
        else:
            logging.info("Граф знаний инициализирован.")

    def save(self):
        nx.write_gml(self.graph, self.save_path)
        config["graph_nodes"] = self.graph.number_of_nodes()
        save_config(config)

    def extract_triples(self, text):
        triples = []
        sentences = re.split(r'[.!?]', text)
        for sent in sentences:
            sent = sent.strip().lower()
            for pattern, rel in self.patterns:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match:
                    subj, obj = match.groups()
                    triples.append((subj.strip(), rel, obj.strip()))
        return triples

    def add_knowledge(self, text):
        triples = self.extract_triples(text)
        for subj, rel, obj in triples:
            self.graph.add_edge(subj, obj, relation=rel)
        if triples:
            logging.info(f"Граф: добавлено {len(triples)} связей.")

    def get_context(self, query, depth=2):
        query = query.lower()
        # Поиск ближайших узлов
        close_nodes = [n for n in self.graph.nodes if Levenshtein.distance(query, n.lower()) < 3]
        nodes = [query] + close_nodes
        context = []
        for node in nodes:
            for neighbor in nx.neighbors(self.graph, node):
                edge = self.graph[node][neighbor]
                context.append(f"{node} {edge['relation']} {neighbor}")
        return context[:5]


# =============================================
# 🔹 МОДЕЛЬ С НЕПРЕРЫВНЫМ ОБУЧЕНИЕМ
# =============================================

class SinModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = os.path.join(MODEL_DIR, "sin_model.bin")
        self.load_or_init()

    def load_or_init(self):
        try:
            if os.path.exists(self.model_path):
                self.model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
                self.tokenizer = AdaptiveTokenizer().tokenizer
                self.model.resize_token_embeddings(len(self.tokenizer))
                logging.info("Модель Sin загружена.")
            else:
                self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
                self.tokenizer = AdaptiveTokenizer().tokenizer
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.save()
                logging.info("Новая модель Sin инициализирована.")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self.tokenizer = AdaptiveTokenizer().tokenizer
            self.model.resize_token_embeddings(len(self.tokenizer))

    def save(self):
        self.model.save_pretrained(MODEL_DIR)
        logging.info("Модель Sin сохранена.")

    def fine_tune(self, texts, epochs=1):
        from datasets import Dataset
        def preprocess(text):
            return " ".join(text.lower().split()[:512])

        cleaned_texts = [preprocess(t) for t in texts if len(t.strip()) > 10]
        if len(cleaned_texts) == 0:
            return

        dataset = Dataset.from_dict({"text": cleaned_texts})
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=1,
            logging_dir=LOGS_DIR,
            logging_steps=100,
            report_to=[],
            no_cuda=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets,
        )

        trainer.train()
        self.save()

        config["learning_progress"] = min(100.0, config["learning_progress"] + len(texts) * 0.15)
        config["total_tokens_seen"] += sum(len(self.tokenizer.encode(t)) for t in cleaned_texts)
        config["unique_concepts"] = len(set(word for t in cleaned_texts for word in t.split()))
        config["last_trained"] = datetime.now().isoformat()
        save_config(config)


# =============================================
# 🔹 ПАРСИНГ И ОЧИСТКА
# =============================================

def scrape_url(url):
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        if article.text and len(article.text) > 100:
            return article.text

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for elem in soup(["script", "style", "nav", "footer", "header", "aside"]):
            elem.decompose()
        return " ".join(soup.stripped_strings)[:10000]
    except Exception as e:
        logging.error(f"Ошибка парсинга URL {url}: {e}")
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
            import docx
            doc = docx.Document(filepath)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif ext == ".pdf":
            import PyPDF2
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return " ".join(page.extract_text() or "" for page in reader.pages)
        else:
            return ""
    except Exception as e:
        logging.error(f"Ошибка чтения файла {filepath}: {e}")
        return ""


# =============================================
# 🔹 ОБУЧЕНИЕ ИЗ ИСТОЧНИКОВ
# =============================================

def learn_from_url(url):
    text = scrape_url(url)
    if len(text) < 50:
        print("❌ Не удалось извлечь текст с URL.")
        return
    save_path = os.path.join(DATA_DIR, "raw", "scraped", f"{hash(url)}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    sin.fine_tune([text])
    memory.add(text, source=f"url:{url}")
    kg.add_knowledge(text)
    memory.save()
    kg.save()
    print("✅ Обучение с URL завершено.")


def learn_from_file(filepath):
    text = read_file(filepath)
    if len(text) < 50:
        print("❌ Не удалось прочитать файл или он пуст.")
        return
    filename = os.path.basename(filepath)
    save_path = os.path.join(DATA_DIR, "raw", "uploaded", filename + ".txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    sin.fine_tune([text])
    memory.add(text, source=f"file:{filename}")
    kg.add_knowledge(text)
    memory.save()
    kg.save()
    print("✅ Обучение с файла завершено.")


# =============================================
# 🔹 ЧАТ С RAG + ГРАФОМ
# =============================================

def chat():
    print("\n" + "="*50)
    print("💬 Чат с Sin. Введите 'выход', 'обучение', 'url', 'файл' для управления.")
    print("="*50)

    dialogue_history = []

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
            learn_from_url(url)
            continue
        elif user_input.lower() == "файл":
            path = input("Путь к файлу (.txt, .docx, .pdf): ").strip()
            if os.path.exists(path):
                learn_from_file(path)
            else:
                print("❌ Файл не найден.")
            continue

        # RAG: поиск знаний
        retrieved = memory.retrieve(user_input, k=2)
        graph_context = kg.get_context(user_input)

        context = ""
        if retrieved:
            context += "[MEM] " + " ".join(retrieved[:2]) + " "
        if graph_context:
            context += "[GRAPH] " + " | ".join(graph_context) + " "

        # Генерация
        prompt = f"{context}[USER] {user_input} [SIN]"
        inputs = sin.tokenizer.encode(prompt, return_tensors="pt")

        outputs = sin.model.generate(
            inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.9,
            top_k=50,
            pad_token_id=sin.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True
        )
        response = sin.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        if not response:
            response = "Я пока не знаю, но запомню это."

        print(f"[Sin]: {response}")

        # Сохраняем диалог
        log_path = os.path.join(CHAT_DIR, "dialogues.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | {user_input} || {response}\n")

        # Дообучение
        sin.fine_tune([f"[USER] {user_input}", f"[SIN] {response}"], epochs=0.3)
        memory.add(user_input, source="chat")
        memory.add(response, source="chat")
        kg.add_knowledge(user_input)
        kg.add_knowledge(response)
        memory.save()
        kg.save()

    sin.save()


def show_learning_progress():
    progress = config["learning_progress"]
    tokens = config["total_tokens_seen"]
    concepts = config["unique_concepts"]
    mem = config["memory_entries"]
    nodes = config["graph_nodes"]

    bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
    color = "🟢" if progress > 70 else "🟡" if progress > 40 else "🔴"

    print("\n📊 **Статус обучения Sin**")
    print(f"{color} Прогресс: {progress:.1f}% [{bar}]")
    print(f"📚 Уникальных понятий: {concepts}")
    print(f"🔖 Всего токенов: {tokens:,}")
    print(f"🧠 Записей в памяти: {mem}")
    print(f"🕸️  Узлов в графе: {nodes}")
    if config["last_trained"]:
        print(f"🕐 Последнее обучение: {config['last_trained'][:16]}")


# =============================================
# 🔹 ОСНОВНОЕ МЕНЮ
# =============================================

def main():
    global sin, memory, kg
    print("🧠 Загрузка Sin — имитации когнитивной модели...")
    sin = SinModel()
    memory = VectorMemory()
    kg = KnowledgeGraph()
    tokenizer = AdaptiveTokenizer()

    print(f"✅ Sin готова. Уровень обучения: {config['learning_progress']:.1f}%")
    print("Введите 'помощь' для списка команд.")

    while True:
        cmd = input("\n> ").strip().lower()
        if cmd in ["помощь", "help"]:
            print("""
Доступные команды:
- чат        — начать диалог с Sin
- обучение   — показать прогресс обучения
- url        — обучить по ссылке
- файл       — обучить по файлу
- выход      — выйти
            """)
        elif cmd == "чат":
            chat()
        elif cmd == "обучение":
            show_learning_progress()
        elif cmd == "url":
            url = input("URL: ").strip()
            learn_from_url(url)
        elif cmd == "файл":
            path = input("Путь к файлу: ").strip()
            if os.path.exists(path):
                learn_from_file(path)
            else:
                print("❌ Файл не найден.")
        elif cmd == "выход":
            sin.save()
            memory.save()
            kg.save()
            save_config(config)
            print("👋 Sin сохранена. До встречи.")
            break
        else:
            print("Неизвестная команда. Введите 'помощь'.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        print(f"Ошибка: {e}")
