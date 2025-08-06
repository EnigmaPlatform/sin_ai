import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
import faiss
import random
import time
import threading
import os
import logging
import pickle
from collections import defaultdict, deque
from gensim.models import KeyedVectors
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from dataclasses import dataclass, field
import hashlib
import requests
import gzip
import shutil
from tqdm import tqdm
import psutil
import pymorphy3
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any, Set
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings

# === Отключение предупреждения о symlinks в Hugging Face ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === НАСТРОЙКИ ПУТЕЙ — ТОЛЬКО В C:\Users\User\Downloads\ ===
BASE_PATH = r"C:\Users\User\Downloads"
EMBEDDING_PATH = os.path.join(BASE_PATH, "cc.ru.300.vec")
EMBEDDING_GZ_PATH = EMBEDDING_PATH + ".gz"
EMBEDDING_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
BIN_PATH = os.path.join(BASE_PATH, "cc.ru.300.bin")
PERSIST_FILE = os.path.join(BASE_PATH, "sin_state.pkl")
GRAPH_FILE = os.path.join(BASE_PATH, "knowledge_graph.pkl")
CHROMA_DIR = os.path.join(BASE_PATH, "chroma_db")
LOG_FILE = os.path.join(BASE_PATH, "sin.log")
TELEGRAM_TOKEN = "7990254673:AAE-7UGlXLWnQ-Dn5D2uyrz0RYDJnBZZKM8"
SUBCONSCIOUS_MODEL = "ai-forever/rugpt3small_based_on_gpt2"
SUBCONSCIOUS_SAVE_DIR = os.path.join(BASE_PATH, "subconscious")

# === ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ===
MAX_NODES = 10000
SLEEP_CYCLE = 15
SAVE_INTERVAL = 1800
MEMORY_HISTORY_LIMIT = 1000
MAX_CONTEXT_LENGTH = 10
MAX_TEXT_LENGTH = 500
UNDERSTANDING_THRESHOLD = 0.65
WORKING_MEMORY_SIZE = 10
INDEX_DIM = 300
ATTENTION_DECAY = 0.93
GENERATION_TEMP = 0.7
DISSONANCE_THRESHOLD = 0.4
FORGET_THRESHOLD = 0.1
RL_REWARD_CORRECT = 2.0
RL_PENALTY_WRONG = -1.0

# === ЛОГГИРОВАНИЕ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIN")

# === МОРФОЛОГИЯ ===
morph = pymorphy3.MorphAnalyzer()

def normalize_word(word: str) -> str:
    word_clean = word.lower().strip(".,!?\"'()[]{}:;—-")
    if not word_clean:
        return ""
    parsed = morph.parse(word_clean)
    return parsed[0].normal_form if parsed else word_clean

# === ФУНКЦИИ ДЛЯ СКАЧИВАНИЯ ===
def calculate_md5(filepath):
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Ошибка при вычислении MD5: {e}")
        return None

def download_embeddings(url, gz_path):
    logger.info(f"Начинаю загрузку с {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(gz_path, 'wb') as f, tqdm(
            desc="📥 Загрузка",
            total=total_size,
            unit='B',
            unit_scale=True,
            colour='green'
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        logger.info(f"✅ Файл сохранён: {gz_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке: {str(e)}")
        return False

def extract_gz(gz_path, vec_path):
    logger.info("🌀 Распаковка архива...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(vec_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"✅ Файл распакован: {vec_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при распаковке: {str(e)}")
        return False

def check_and_download_embeddings():
    if os.path.exists(EMBEDDING_PATH):
        logger.info(f"✅ Файл найден: {EMBEDDING_PATH}")
        return True
    logger.warning(f"❌ Файл не найден: {EMBEDDING_PATH}")
    print("Файл эмбеддингов отсутствует. Начать загрузку? (y/n): ", end="")
    if input().lower() != 'y':
        logger.critical("❌ Загрузка отменена.")
        return False
    if not download_embeddings(EMBEDDING_URL, EMBEDDING_GZ_PATH):
        logger.critical("❌ Не удалось загрузить файл.")
        return False
    if not extract_gz(EMBEDDING_GZ_PATH, EMBEDDING_PATH):
        logger.critical("❌ Не удалось распаковать файл.")
        return False
    logger.info("✅ Эмбеддинги готовы!")
    return True

# === RuEmbedder ===
class RuEmbedder:
    def __init__(self, filepath=EMBEDDING_PATH):
        if not check_and_download_embeddings():
            raise RuntimeError("Не удалось подготовить эмбеддинги.")
        if os.path.exists(BIN_PATH):
            logger.info("🌀 Загрузка из бинарного файла (быстро)...")
            self.model = KeyedVectors.load(BIN_PATH)
        else:
            logger.info("🌀 Загрузка из текстового файла (ограничено 50k слов)...")
            total_lines = 500000 + 1
            with tqdm(desc="🧠 Загрузка слов", total=total_lines, colour='blue') as pbar:
                self.model = KeyedVectors.load_word2vec_format(filepath, binary=False, limit=50000)
                for _ in range(total_lines):
                    pbar.update(1)
            logger.info("💾 Сохранение в бинарный формат для будущих запусков...")
            self.model.save(BIN_PATH)
        self.dim = self.model.vector_size
        logger.info(f"✅ Загружено {len(self.model.key_to_index)} слов (dim={self.dim})")

    def get_vector(self, word: str) -> np.ndarray:
        norm = normalize_word(word)
        if not norm or norm not in self.model:
            return np.random.normal(0, 0.1, self.dim)
        return self.model[norm].copy()

# === SubconsciousModule — LLM как "подсознание" ===
class SubconsciousModule:
    def __init__(self, model_name=SUBCONSCIOUS_MODEL, save_dir=SUBCONSCIOUS_SAVE_DIR):
        self.model_name = model_name
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Сначала пытаемся загрузить из папки, иначе — из HF, потом сохраняем."""
        os.makedirs(self.save_dir, exist_ok=True)
        # 1. Проверяем, есть ли файлы в папке
        if self._is_model_saved():
            logger.info(f"📥 Попытка загрузить подсознание из: {self.save_dir}")
            if self._try_load_from_disk():
                logger.info(f"✅ Подсознание успешно загружено из: {self.save_dir}")
                return
        # 2. Если не получилось — загружаем из Hugging Face
        logger.info(f"🌐 Загрузка подсознания из Hugging Face: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            logger.info(f"🧠 Подсознание загружено из Hugging Face: {self.model_name}")
            # 3. Сразу сохраняем локально
            self._save_to_disk()
            logger.info(f"💾 Подсознание сохранено в: {self.save_dir}")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели из HF: {e}")
            raise

    def _is_model_saved(self) -> bool:
        required = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        return all(os.path.exists(os.path.join(self.save_dir, f)) for f in required)

    def _try_load_from_disk(self) -> bool:
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.save_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            self.model.to(self.device)
            self.model.eval()
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке с диска: {e}")
            return False

    def _save_to_disk(self):
        try:
            self.model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)
            logger.info(f"💾 Подсознание сохранено: {self.save_dir}")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении подсознания: {e}")

    def generate(self, prompt: str, max_length: int = 100) -> str:
        try:
            outputs = self.generator(
                prompt,
                max_length=max_length,
                temperature=GENERATION_TEMP,
                top_k=50,
                do_sample=True,
                num_return_sequences=1
            )
            return outputs[0]['generated_text'].replace(prompt, "").strip()
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return "Я пока не могу ответить."

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.base_model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

# === СТРУКТУРЫ ПАМЯТИ ===
@dataclass
class MemoryItem:
    vector: List[float]
    text: str
    level: int
    timestamp: float
    access_count: int = 0
    phase_cluster_id: Optional[int] = None
    reward_score: float = 0.0
    coherence_score: float = 1.0

    @staticmethod
    def from_np(vector: np.ndarray, text: str, level: int, timestamp: float, **kwargs):
        return MemoryItem(
            vector=vector.tolist(),
            text=text,
            level=level,
            timestamp=timestamp,
            **kwargs
        )

    def to_np_vector(self) -> np.ndarray:
        return np.array(self.vector)

@dataclass
class SemanticEpisode:
    slots: Dict[str, str]
    frame_type: Optional[str]
    text: str
    vector: np.ndarray
    timestamp: float
    coherence_score: float = 1.0
    prediction_error: float = 0.0

    def __post_init__(self):
        if isinstance(self.vector, list):
            self.vector = np.array(self.vector)
        elif self.vector is None:
            self.vector = np.zeros(300)

@dataclass
class Scene:
    episodes: List[SemanticEpisode]
    theme: str
    start_time: float
    end_time: float
    relevance_score: float = 0.0

@dataclass
class Plot:
    scenes: List[Scene]
    title: str
    coherence: float

@dataclass
class DialogContext:
    last_messages: List[str] = None
    timestamps: List[float] = None
    current_theme: Optional[str] = None
    thematic_attention: float = 0.0

    def __post_init__(self):
        if self.last_messages is None:
            self.last_messages = []
        if self.timestamps is None:
            self.timestamps = []

    def add_message(self, message: str):
        self.last_messages.append(message)
        self.timestamps.append(time.time())
        if len(self.last_messages) > MAX_CONTEXT_LENGTH:
            self.last_messages.pop(0)
            self.timestamps.pop(0)

@dataclass
class Goal:
    description: str
    priority: float
    steps: List[str]
    achieved: bool = False
    created_at: float = field(default_factory=time.time)

@dataclass
class Emotion:
    name: str
    intensity: float
    decay: float = 0.01
    trigger_threshold: float = 0.7

# === Resonator ===
class Resonator:
    __slots__ = ['id', 'freq', 'phase', 'amplitude', 'damping', 'connections',
                 'pattern', 'level', 'last_activation', 'attention', 'phase_history',
                 'sin_instance']

    def __init__(self, node_id: int, level: int = 0):
        self.id = node_id
        self.freq = 1.0
        self.phase = 0.0
        self.amplitude = 0.0
        self.damping = 0.1
        self.connections = defaultdict(float)
        self.pattern = None
        self.level = level
        self.last_activation = 0.0
        self.attention = 1.0
        self.phase_history = deque(maxlen=100)
        self.sin_instance = None

    def excite(self, amp: float, phase_offset: float = 0.0):
        self.amplitude = amp * self.attention
        self.phase = phase_offset
        self.last_activation = amp

    def step(self, dt: float = 0.1):
        if self.amplitude > 0.01:
            self.phase += self.freq * dt
            self.phase %= (2 * np.pi)
            self.amplitude *= (1 - self.damping * dt)
            self.attention *= ATTENTION_DECAY
            self.phase_history.append(self.phase)
            if self.sin_instance:
                activation_row = [0.0] * len(self.sin_instance.nodes)
                idx = list(self.sin_instance.nodes.keys()).index(self.id)
                activation_row[idx] = self.amplitude
                self.sin_instance.activation_history.append(activation_row)
        else:
            self.amplitude = 0.0

# === Hippocampus ===
class Hippocampus:
    def __init__(self, capacity: int = WORKING_MEMORY_SIZE):
        self.working_memory = deque(maxlen=capacity)
        self.consolidation_threshold = 0.7
        self.predicted_items = []

    def add(self, item: MemoryItem):
        self.working_memory.append(item)

    def add_prediction(self, item: MemoryItem):
        self.predicted_items.append(item)

    def get_prediction_error(self, actual_item: MemoryItem) -> float:
        if not self.predicted_items:
            return 1.0
        predicted = self.predicted_items[-1]
        sim = cosine_similarity([actual_item.to_np_vector()], [predicted.to_np_vector()])[0][0]
        return 1.0 - sim

    def consolidate(self, long_term_memory: list, vector_index, prediction_error: float = 0.0):
        consolidated_count = 0
        for item in self.working_memory:
            if prediction_error < 0.3:
                item.coherence_score = min(1.0, item.coherence_score + 0.1)
            if item.coherence_score > self.consolidation_threshold or prediction_error < 0.5:
                long_term_memory.append(item)
                vector_index.add_vector(item.to_np_vector(), len(long_term_memory) - 1)
                consolidated_count += 1
        logger.debug(f"🧠 Консолидировано {consolidated_count} элементов.")
        self.working_memory.clear()
        self.predicted_items.clear()

# === VectorIndex ===
class VectorIndex:
    def __init__(self, dim: int = INDEX_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []

    def add_vector(self, vector: np.ndarray, external_id: int):
        vec = np.array([vector], dtype=np.float32)
        self.index.add(vec)
        self.vectors.append(external_id)

    def search_similar(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        if self.index.ntotal == 0:
            return []
        query = np.array([query], dtype=np.float32)
        distances, indices = self.index.search(query, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.vectors):
                similarity = 1 / (1 + dist)
                results.append((similarity, self.vectors[idx]))
        return results

# === AdaptiveParams ===
class AdaptiveParams:
    def __init__(self):
        self.forget_threshold = 0.1
        self.dissonance_threshold = 0.4
        self.rl_learning_rate = 0.1
        self.reward_history = deque(maxlen=100)

    def update(self, reward: float):
        self.reward_history.append(reward)
        avg = np.mean(self.reward_history) if self.reward_history else 0.0
        self.forget_threshold = 0.1 + 0.1 * avg
        self.dissonance_threshold = 0.4 + 0.2 * avg
        logger.debug(f"⚙️ Параметры адаптированы. Forget: {self.forget_threshold:.3f}, Dissonance: {self.dissonance_threshold:.3f}, Avg Reward: {avg:.3f}")

# === KnowledgeGraph ===
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concepts = {}

    def add_concept(self, name: str, vector: np.ndarray, parents: List[str] = None, children: List[str] = None, relations: Dict[str, List[str]] = None):
        if parents is None: parents = []
        if children is None: children = []
        if relations is None: relations = {}
        self.graph.add_node(name)
        self.concepts[name] = vector
        for parent in parents:
            self.graph.add_edge(parent, name, relation="hypernym")
        for child in children:
            self.graph.add_edge(name, child, relation="hyponym")
        for rel_type, targets in relations.items():
            for target in targets:
                self.graph.add_edge(name, target, relation=rel_type)
        logger.debug(f"📊 Добавлен концепт '{name}' в граф знаний.")

    def find_path(self, source: str, target: str) -> List[str]:
        try:
            return nx.shortest_path(self.graph, source, target)
        except:
            return []

    def get_neighbors(self, concept: str, depth=1) -> List[str]:
        if not self.graph.has_node(concept):
            return []
        neighbors = list(self.graph.neighbors(concept))
        if depth > 1:
            for neighbor in list(neighbors):
                neighbors.extend(self.get_neighbors(neighbor, depth-1))
        return list(set(neighbors))

    def get_relations(self, concept: str) -> Dict[str, List[str]]:
        if not self.graph.has_node(concept):
            return {}
        relations = defaultdict(list)
        for _, target, data in self.graph.out_edges(concept, data=True):
            rel_type = data.get('relation', 'unknown')
            relations[rel_type].append(target)
        return dict(relations)

    def save_graph(self):
        try:
            with open(GRAPH_FILE, 'wb') as f:
                pickle.dump({'graph': self.graph, 'concepts': self.concepts}, f)
            logger.info(f"📊 Граф знаний сохранён: {GRAPH_FILE}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения графа: {e}")

    def load_graph(self):
        if not os.path.exists(GRAPH_FILE):
            return
        try:
            with open(GRAPH_FILE, 'rb') as f:
                data = pickle.load(f)
            self.graph = data['graph']
            self.concepts = data['concepts']
            logger.info(f"📊 Граф знаний загружен из {GRAPH_FILE}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки графа: {e}")

# === SemanticMemory с ChromaDB ===
class SemanticMemory:
    def __init__(self, embedder, chroma_dir=CHROMA_DIR):
        self.embedder = embedder
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="semantic_episodes")
        self.episodes: List[SemanticEpisode] = []
        self.scenes: List[Scene] = []
        self.clusters = {}
        logger.info(f"💾 ChromaDB инициализирована: {chroma_dir}")

    def add_episode(self, text: str, slots: Dict[str, str] = None, frame_type: str = None):
        if slots is None:
            slots = {}
        words = [w for w in text.split() if w.isalpha()]
        if not words:
            return
        vec = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
        episode = SemanticEpisode(
            slots=slots,
            frame_type=frame_type,
            text=text,
            vector=vec,
            timestamp=time.time()
        )
        self.collection.add(
            embeddings=[vec.tolist()],
            documents=[text],
            metadatas=[{"frame": frame_type or "unknown", "timestamp": str(time.time())}],
            ids=[f"ep_{len(self.episodes)}"]
        )
        self.episodes.append(episode)
        logger.debug(f"🧠 Эпизод добавлен в Chroma: '{text}'")

    def search_by_semantics(self, query: str, n_results: int = 5) -> List[SemanticEpisode]:
        query_vec = np.mean([self.embedder.get_vector(w) for w in query.split()], axis=0)
        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n_results
        )
        matches = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            for ep in self.episodes:
                if ep.text == doc:
                    matches.append(ep)
                    break
        return matches

    def cluster_episodes(self, n_clusters: int = 5):
        if len(self.episodes) < n_clusters:
            return
        vectors = np.array([ep.vector for ep in self.episodes])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)
        self.clusters = {}
        for i, label in enumerate(labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(self.episodes[i])
        logger.info(f"🎯 Сформировано {n_clusters} кластеров по темам.")

    def form_scenes_from_clusters(self):
        self.scenes = []
        for cluster_id, episodes in self.clusters.items():
            if len(episodes) < 2:
                continue
            theme = self._infer_theme(episodes)
            start_time = min(ep.timestamp for ep in episodes)
            end_time = max(ep.timestamp for ep in episodes)
            scene = Scene(episodes=episodes, theme=theme, start_time=start_time, end_time=end_time)
            self.scenes.append(scene)
            logger.info(f"🎬 Сцена сформирована: '{theme}' ({len(episodes)} эпизодов)")

    def _infer_theme(self, episodes: List[SemanticEpisode]) -> str:
        all_words = " ".join([ep.text for ep in episodes])
        words = [w for w in all_words.split() if len(w) > 3]
        if not words:
            return "общее"
        word_freq = defaultdict(int)
        for w in words:
            word_freq[w] += 1
        theme = max(word_freq, key=word_freq.get)
        return theme

    def visualize_clusters(self):
        if not self.clusters:
            logger.warning("Нет кластеров для визуализации.")
            return
        vectors = np.array([ep.vector for ep in self.episodes])
        labels = np.array([lbl for lbl, eps in self.clusters.items() for _ in eps])
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(vectors)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title("Кластеры эпизодов (t-SNE)")
        plt.xlabel("Компонента 1")
        plt.ylabel("Компонента 2")
        plt.tight_layout()
        plt.show()

# === AutonomousLearner ===
class AutonomousLearner:
    def __init__(self, sin_instance):
        self.sin = sin_instance
        self.is_learning = False
        self.learning_thread = None
        self.log = []

    def log_event(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log.append(log_entry)
        logger.info(f"🤖 АВТООБУЧЕНИЕ: {message}")

    def start_autonomous_learning(self, duration_minutes: int = 30):
        if self.is_learning:
            return "Обучение уже запущено"
        self.is_learning = True
        self.log = []
        self.log_event(f"Запуск автономного обучения на {duration_minutes} минут")

        def learning_loop():
            end_time = time.time() + duration_minutes * 60
            cycle = 0
            while time.time() < end_time and self.is_learning:
                cycle += 1
                self.log_event(f"🔄 Автономный цикл {cycle}")
                self.generate_curiosity_questions()
                self.learn_from_urls()
                self.self_reflect()
                self.strengthen_connections()
                self.detect_conflicts()
                self.form_and_test_hypotheses()

                if cycle % 6 == 0:
                    self.log_event("Шаг: Автосохранение...")
                    self.sin.save_state()
                    self.sin.knowledge_graph.save_graph()
                    self.sin.semantic_memory.cluster_episodes()
                    self.sin.semantic_memory.form_scenes_from_clusters()

                self.log_event(f"Завершён цикл {cycle}. Пауза на 5 минут.")
                time.sleep(300)

            self.is_learning = False
            self.log_event("✅ Автономное обучение завершено")

        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()
        return f"Обучение запущено на {duration_minutes} минут"

    def stop_learning(self):
        if self.is_learning:
            self.log_event("🛑 Получен сигнал остановки.")
            self.is_learning = False
            if self.learning_thread:
                self.learning_thread.join(timeout=1)
            self.log_event("🛑 Автономное обучение остановлено")

    def generate_curiosity_questions(self):
        if len(self.sin.memory) < 2:
            return
        concepts = list(self.sin.knowledge_graph.graph.nodes())
        if len(concepts) < 2:
            return
        concept = random.choice(concepts)
        question = f"Что я знаю о {concept}? А что ещё не знаю?"
        self.sin.pending_questions.append(question)
        self.log_event(f"❓ Сгенерирован вопрос любопытства: {question}")

    def learn_from_urls(self):
        urls = [
            "https://ru.wikipedia.org/wiki/Искусственный_интеллект",
            "https://habr.com/ru/news/"
        ]
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 50][:10]
                for sent in sentences:
                    self.sin.learn(sent[:500], user_feedback="good")
                self.log_event(f"✅ Обучение на URL завершено: {url}")
            except Exception as e:
                logger.error(f"❌ Ошибка при обучении на URL {url}: {e}")

    def self_reflect(self):
        if not self.sin.memory:
            return
        recent = self.sin.memory[-5:]
        topics = [m.text for m in recent if len(m.text) > 3]
        if topics:
            reflection = f"Я недавно изучал: {', '.join(topics[:3])}. Что это может значить?"
            self.sin.pending_questions.append(reflection)
            self.log_event(f"🧠 Саморефлексия: {reflection}")

    def strengthen_connections(self):
        strengthened_count = 0
        for mem in self.sin.memory:
            if mem.access_count > 5:
                old_score = mem.coherence_score
                mem.coherence_score = min(1.0, mem.coherence_score + 0.1)
                if mem.coherence_score > old_score:
                    strengthened_count += 1
        self.log_event(f"💪 Укреплено {strengthened_count} связей в памяти.")

    def detect_conflicts(self):
        conflicts_found = 0
        for i, mem1 in enumerate(self.sin.memory):
            for j, mem2 in enumerate(self.sin.memory[i+1:], i+1):
                sim = cosine_similarity([mem1.to_np_vector()], [mem2.to_np_vector()])[0][0]
                if sim > 0.8 and ("не " in mem1.text) != ("не " in mem2.text):
                    conflict = f"Обнаружено противоречие: '{mem1.text}' vs '{mem2.text}' (схожесть: {sim:.2f})"
                    logger.warning(f"🤖 АВТООБУЧЕНИЕ: {conflict}")
                    self.sin.pending_questions.append(f"Я нашёл противоречие: {mem1.text} и {mem2.text}. Какое утверждение верно?")
                    conflicts_found += 1
        self.log_event(f"🔍 Обнаружено {conflicts_found} конфликтов.")

    def form_and_test_hypotheses(self):
        concepts = list(self.sin.knowledge_graph.graph.nodes())
        if len(concepts) < 3:
            self.log_event("Недостаточно концептов в графе для формирования гипотез.")
            return
        cause = random.choice(concepts)
        neighbors = list(self.sin.knowledge_graph.graph.neighbors(cause))
        if not neighbors:
            return
        effect = random.choice(neighbors)
        other_concepts = [c for c in concepts if c != cause and c != effect and not self.sin.knowledge_graph.graph.has_edge(cause, c)]
        if not other_concepts:
            return
        test_concept = random.choice(other_concepts)
        hypothesis = f"Если '{cause}' приводит к '{effect}', то это похоже на '{test_concept}'?"
        self.sin.pending_questions.append(hypothesis)
        self.log_event(f"🧠 Сформирована гипотеза: {hypothesis}")

    def get_log(self) -> List[str]:
        return self.log.copy()

# === MultiAgentSystem ===
class MultiAgentSystem:
    def __init__(self, base_sin):
        self.agents = {"main": base_sin}
        self.communication_history = deque(maxlen=100)

    def create_agent(self, name: str):
        new_sin = Sin(persist_file=os.path.join(BASE_PATH, f"sin_state_{name}.pkl"))
        self.agents[name] = new_sin
        logger.info(f"🤖 Создан агент: {name}")
        return new_sin

    def communicate(self, sender: str, receiver: str, message: str) -> str:
        if sender not in self.agents or receiver not in self.agents:
            return "Агент не найден"
        sender_agent = self.agents[sender]
        receiver_agent = self.agents[receiver]
        sender_agent.learn(message)
        response = receiver_agent.respond(message)
        self.communication_history.append({
            "sender": sender, "receiver": receiver, "message": message, "response": response, "timestamp": time.time()
        })
        return response

    def debate(self, topic: str, agents: List[str]) -> List[str]:
        if not all(a in self.agents for a in agents):
            return ["Один из агентов не найден"]
        results = []
        for agent in agents:
            response = self.agents[agent].respond(topic)
            results.append(f"{agent}: {response}")
        return results

# === SIN — ОСНОВНАЯ СИСТЕМА ===
class Sin:
    VERSION = "19.5"

    def __init__(self, persist_file: str = PERSIST_FILE):
        os.makedirs(SUBCONSCIOUS_SAVE_DIR, exist_ok=True)
        self.embedder = RuEmbedder()
        self.subconscious = SubconsciousModule()
        self.nodes = {}
        self.node_counter = 0
        self.memory = []
        self.activation_history = deque(maxlen=100)
        self.t = 0
        self.sleeping = False
        self.word_frequency = defaultdict(int)
        self.pending_questions = []
        self.dialog_context = DialogContext()
        self.persist_file = persist_file
        self.last_save_time = time.time()
        self.rl_policy = {"ask_question": 0.7, "generate": 0.5}
        self.phase_clusters = []
        self.hippocampus = Hippocampus()
        self.params = AdaptiveParams()
        self.vector_index = VectorIndex(dim=self.embedder.dim)
        self.cognitive_load = 0.0
        self.level_nodes = [[] for _ in range(3)]
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.load_graph()
        self.semantic_memory = SemanticMemory(self.embedder)
        self.goals = []
        self.emotions = {
            "curiosity": Emotion("curiosity", 0.5),
            "certainty": Emotion("certainty", 0.7),
            "social_connection": Emotion("social_connection", 0.3)
        }
        self.autonomous_learner = AutonomousLearner(self)
        self.multi_agent_system = MultiAgentSystem(self)
        self._init_system()
        logger.info(f"🌐 SIN v{self.VERSION} запущен. Узлов: {len(self.nodes)}, Память: {len(self.memory)}")
        self.auto_pretrain()

    def auto_pretrain(self):
        logger.info("🚀 Начинаем автопредобучение...")
        for word in list(self.embedder.model.key_to_index.keys())[:1000]:
            self.learn(word, user_feedback="good")
        urls = [
            "https://ru.wikipedia.org/wiki/Искусственный_интеллект",
            "https://habr.com/ru/news/"
        ]
        for url in urls:
            self.learn_from_url(url)
        logger.info("✅ Автопредобучение завершено.")

    def learn_from_url(self, url: str):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 50][:10]
            for sent in sentences:
                self.learn(sent[:500], user_feedback="good")
            logger.info(f"✅ Обучение на URL завершено: {url}")
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении на URL {url}: {e}")

    def _init_system(self):
        if os.path.exists(self.persist_file):
            self._load_state()
            logger.info("💾 Состояние загружено")
        else:
            logger.info("🆕 Создана новая модель")

    def _load_state(self):
        try:
            with open(self.persist_file, 'rb') as f:
                data = pickle.load(f)
            self.node_counter = data.get('node_counter', 0)
            self.t = data.get('t', 0)
            self.word_frequency = defaultdict(int, data.get('word_frequency', {}))
            self.rl_policy = data.get('rl_policy', {"ask_question": 0.7, "generate": 0.5})
            self.phase_clusters = data.get('phase_clusters', [])
            self.dialog_context = DialogContext(**data.get('dialog_context', {}))
            self.params = data.get('params', AdaptiveParams())
            self.memory = [MemoryItem(**item) for item in data.get('memory', [])]
            for item in self.memory:
                vec = np.array(item.vector, dtype=np.float32)
                self.vector_index.add_vector(vec, len(self.memory) - 1)
            self.goals = [Goal(**g) for g in data.get('goals', [])]
            self.emotions = data.get('emotions', self.emotions)
            logger.info(f"💾 Состояние загружено из {self.persist_file}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")

    def save_state(self):
        try:
            serializable = {
                'nodes': self.nodes,
                'node_counter': self.node_counter,
                'memory': [item.__dict__ for item in self.memory],
                't': self.t,
                'word_frequency': dict(self.word_frequency),
                'rl_policy': self.rl_policy,
                'phase_clusters': self.phase_clusters,
                'dialog_context': {
                    'last_messages': self.dialog_context.last_messages,
                    'timestamps': self.dialog_context.timestamps,
                    'current_theme': self.dialog_context.current_theme,
                    'thematic_attention': self.dialog_context.thematic_attention
                },
                'params': self.params,
                'goals': [g.__dict__ for g in self.goals],
                'emotions': self.emotions
            }
            with open(self.persist_file, 'wb') as f:
                pickle.dump(serializable, f)
            self.knowledge_graph.save_graph()
            self.subconscious._save_to_disk()
            logger.info(f"💾 Сохранено в {self.persist_file}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")

    def _auto_save(self):
        if time.time() - self.last_save_time > SAVE_INTERVAL:
            self.save_state()
            self.last_save_time = time.time()

    def tokenize(self, text: str) -> List[str]:
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        words = [normalize_word(w) for w in text.split() if w.isalpha()]
        for w in words:
            self.word_frequency[w] += 1
        return words

    def _predict_next(self, context_words: List[str]) -> np.ndarray:
        if not context_words:
            return np.random.normal(0, 0.1, self.embedder.dim)
        context_vectors = [self.embedder.get_vector(w) for w in context_words[-3:]]
        avg_context = np.mean(context_vectors, axis=0)
        noise = np.random.normal(0, 0.05, avg_context.shape)
        predicted_vec = avg_context + noise
        return predicted_vec / (np.linalg.norm(predicted_vec) + 1e-8)

    def are_in_phase(self, n1: Resonator, n2: Resonator, tol=0.5) -> bool:
        return abs((n1.phase - n2.phase) % (2 * np.pi)) < tol

    def assign_to_phase_cluster(self, node: Resonator) -> int:
        for cid, cluster in enumerate(self.phase_clusters):
            if cluster and cluster[0] in self.nodes:
                if self.are_in_phase(node, self.nodes[cluster[0]]):
                    cluster.append(node.id)
                    return cid
        new_cl = [node.id]
        self.phase_clusters.append(new_cl)
        return len(self.phase_clusters) - 1

    def conflict_detector(self, new_vec: np.ndarray, threshold=0.8) -> List[str]:
        conflicting = []
        results = self.vector_index.search_similar(new_vec, k=10)
        for sim, idx in results:
            if sim > threshold and "не " in self.memory[idx].text:
                conflicting.append(self.memory[idx].text)
        return conflicting

    def calculate_understanding_score(self, text: str) -> float:
        words = self.tokenize(text)
        if not words:
            return 0.0
        query_vec = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
        results = self.vector_index.search_similar(query_vec, k=20)
        if not results:
            return 0.0
        top_sim = results[0][0]
        now = time.time()
        context_relevance = 0.0
        for msg, ts in zip(self.dialog_context.last_messages, self.dialog_context.timestamps):
            decay = np.exp(-0.1 * (now - ts))
            context_words = self.tokenize(msg)
            if context_words:
                context_vec = np.mean([self.embedder.get_vector(w) for w in context_words], axis=0)
                sim = cosine_similarity([query_vec], [context_vec])[0][0]
                context_relevance += sim * decay
        context_relevance /= max(len(self.dialog_context.last_messages), 1)
        return 0.6 * top_sim + 0.4 * context_relevance

    def learn(self, text: str, user_feedback: str = "neutral") -> Dict:
        self.dialog_context.add_message(text)
        understanding = self.calculate_understanding_score(text)
        words = self.tokenize(text)
        if not words:
            return {"status": "empty", "response": "Пусто", "understanding": understanding}
        context_words = []
        for msg in self.dialog_context.last_messages[-2:]:
            context_words.extend(self.tokenize(msg))
        context_words.extend(words[:-1])
        predicted_vec = self._predict_next(context_words)
        predicted_item = MemoryItem.from_np(predicted_vec, "[предсказание]", level=0, timestamp=time.time())
        self.hippocampus.add_prediction(predicted_item)
        actual_last_word = words[-1]
        actual_vec = self.embedder.get_vector(actual_last_word)
        actual_item = MemoryItem.from_np(actual_vec, actual_last_word, level=0, timestamp=time.time())
        prediction_error = self.hippocampus.get_prediction_error(actual_item)
        logger.debug(f"📈 Ошибка предсказания для '{actual_last_word}': {prediction_error:.3f}")
        total_vec = np.zeros(self.embedder.dim)
        reward = 0.0
        items_to_add = []
        for i, word in enumerate(words):
            vec = self.embedder.get_vector(word)
            total_vec += vec
            conflicts = self.conflict_detector(vec)
            if conflicts:
                reward -= 0.5
            nid = self.node_counter
            node = Resonator(nid)
            node.sin_instance = self
            node.pattern = vec.copy()
            self.nodes[nid] = node
            node.excite(1.0)
            self.node_counter += 1
            cluster_id = self.assign_to_phase_cluster(node)
            mem_item = MemoryItem.from_np(
                vector=vec,
                text=word,
                level=0,
                timestamp=time.time(),
                phase_cluster_id=cluster_id,
                reward_score=reward,
                coherence_score=1.0 - len(conflicts) * 0.3
            )
            items_to_add.append(mem_item)
            if len(conflicts) == 0:
                if len(words) > 1:
                    phrase = ' '.join(words)
                    self.knowledge_graph.add_concept(word, vec, children=[phrase])
                    self.knowledge_graph.add_concept(phrase, total_vec, parents=[word])
                if i > 0:
                    prev_word = words[i-1]
                    self.knowledge_graph.add_concept(prev_word, self.embedder.get_vector(prev_word), relations={"next": [word]})
                    self.knowledge_graph.add_concept(word, vec, relations={"prev": [prev_word]})
        total_vec /= len(words)
        phrase_item = MemoryItem.from_np(
            vector=total_vec,
            text=' '.join(words),
            level=1,
            timestamp=time.time(),
            reward_score=reward
        )
        items_to_add.append(phrase_item)
        for item in items_to_add:
            self.hippocampus.add(item)
        self.hippocampus.consolidate(self.memory, self.vector_index, prediction_error)
        slots = {}
        frame_type = "COMMUNICATION" if any(w in text for w in ["ты", "я", "мы"]) else None
        self.semantic_memory.add_episode(text, slots=slots, frame_type=frame_type)
        if user_feedback == "good":
            reward = RL_REWARD_CORRECT
            self.emotions["certainty"].intensity = min(1.0, self.emotions["certainty"].intensity + 0.1)
        elif user_feedback == "bad":
            reward = RL_PENALTY_WRONG
            self.emotions["certainty"].intensity = max(0.0, self.emotions["certainty"].intensity - 0.2)
        curiosity_boost = 0.05 + 0.1 * prediction_error
        self.emotions["curiosity"].intensity = min(1.0, self.emotions["curiosity"].intensity + curiosity_boost)
        self.params.update(reward)
        old_ask_prob = self.rl_policy["ask_question"]
        self.rl_policy["ask_question"] = 0.3 + 0.4 * (reward / 2.0 if reward != 0 else 0.7) + 0.3 * self.emotions["curiosity"].intensity
        logger.debug(f"⚖️ Политика RL обновлена. Вероятность вопроса: {old_ask_prob:.3f} -> {self.rl_policy['ask_question']:.3f}")
        self._auto_save()
        status = "understood" if understanding > UNDERSTANDING_THRESHOLD else "partially"
        return {
            "status": status,
            "response": f"{'🧠' if understanding > 0.7 else '🤔'} Понял: '{text}' (понимание: {understanding:.2f}, ошибка предсказания: {prediction_error:.2f})",
            "understanding": understanding,
            "prediction_error": prediction_error
        }

    def _extract_concepts_from_text(self, text: str) -> Set[str]:
        words = self.tokenize(text)
        concepts = set()
        for word in words:
            if word in self.knowledge_graph.concepts:
                concepts.add(word)
        logger.debug(f"🧩 Извлечены концепты из '{text}': {concepts}")
        return concepts

    def _generate_response_from_knowledge(self, concepts: Set[str], context_concepts: Set[str]) -> Optional[str]:
        if not concepts:
            return None
        response_parts = []
        for concept in concepts:
            relations = self.knowledge_graph.get_relations(concept)
            for rel_type, targets in relations.items():
                relevant_targets = [t for t in targets if t in context_concepts or not context_concepts]
                if relevant_targets:
                    target = random.choice(relevant_targets)
                    response_parts.append(f"{concept} {rel_type.replace('_', ' ')} {target}.")
        for concept in list(concepts)[:3]:
            neighbors = self.knowledge_graph.get_neighbors(concept, depth=1)
            relevant_neighbors = [n for n in neighbors if n in context_concepts or not context_concepts]
            if relevant_neighbors:
                neighbor = random.choice(relevant_neighbors)
                response_parts.append(f"Я также знаю о {neighbor}, связанном с {concept}.")
        if response_parts:
            random.shuffle(response_parts)
            final_response = " ".join(response_parts[:2])
            logger.debug(f"🗣️ Сгенерирован ответ на основе знаний: {final_response}")
            return final_response
        return None

    def semantic_search(self, query: str) -> str:
        results = self.semantic_memory.search_by_semantics(query)
        if not results:
            return f"🔍 Я не нашёл эпизодов по запросу '{query}'."
        best = results[0]
        return f"🧠 Я помню: '{best.text}' (похоже на '{query}')"

    def respond(self, text: str) -> str:
        if self.sleeping:
            return "Zzz... Sin спит."
        self.dialog_context.add_message(text)
        understanding = self.calculate_understanding_score(text)
        logger.info(f"💬 Получен запрос: '{text}' (понимание: {understanding:.3f})")
        if understanding < 0.2:
            logger.warning("❓ Запрос не понят. Запрашиваю уточнение.")
            return "❓ Совсем новое. Расскажи подробнее."
        words = self.tokenize(text)
        if not words:
            logger.info("🗣️ Пустой запрос.")
            return "Я слушаю..."
        query_vec = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
        query_concepts = self._extract_concepts_from_text(text)
        context_concepts = set()
        for ctx_msg in self.dialog_context.last_messages[-2:]:
            context_concepts.update(self._extract_concepts_from_text(ctx_msg))
        knowledge_response = self._generate_response_from_knowledge(query_concepts, context_concepts)
        if knowledge_response and random.random() < 0.7:
            logger.info("🧠 Ответ сгенерирован на основе графа знаний.")
            return f"🧠 {knowledge_response}"
        semantic_response = self.semantic_search(text)
        if "не нашёл" not in semantic_response:
            logger.info("🔍 Ответ найден через семантический поиск.")
            return semantic_response
        curiosity = self.emotions["curiosity"].intensity
        certainty = self.emotions["certainty"].intensity
        if self.pending_questions and (random.random() < self.rl_policy["ask_question"] * curiosity or curiosity > 0.8):
            question = self.pending_questions.pop(0)
            logger.info(f"❓ Задаю вопрос: {question}")
            return f"❓ {question}"
        prompt = f"Пользователь: {text}\nSin:"
        response = self.subconscious.generate(prompt, max_length=100)
        logger.info("💬 Ответ сгенерирован через подсознание.")
        return f"💬 {response}"

    def dream_cycle(self):
        logger.info("💭 Sin видит сны...")
        dreamed_count = 0
        for _ in range(5):
            if not self.memory:
                continue
            mem = random.choice(self.memory)
            vec = mem.to_np_vector()
            noise = np.random.normal(0, 0.05, vec.shape)
            dream_vec = vec + noise
            dream_vec /= (np.linalg.norm(dream_vec) + 1e-8)
            dream_item = MemoryItem.from_np(
                vector=dream_vec,
                text=f"[сон: {mem.text}]",
                level=mem.level,
                timestamp=time.time()
            )
            self.hippocampus.add(dream_item)
            dreamed_count += 1
            time.sleep(0.3)
        logger.info(f"💭 Сгенерировано {dreamed_count} сонных воспоминаний.")
        self.hippocampus.consolidate(self.memory, self.vector_index)
        self.sleeping = False
        self.cognitive_load *= 0.5
        logger.info("✨ Sin проснулся. Память укреплена.")

    def start_sleep(self):
        self.sleeping = True
        logger.info("🌙 Sin засыпает...")
        threading.Thread(target=self.dream_cycle, daemon=True).start()

    def start_autonomous_learning(self, duration: int = 30):
        return self.autonomous_learner.start_autonomous_learning(duration)

    def stop_autonomous_learning(self):
        self.autonomous_learner.stop_learning()

    def get_autonomous_learning_log(self) -> List[str]:
        return self.autonomous_learner.get_log()

    def status(self):
        understanding_avg = np.mean([self.calculate_understanding_score(m.text) for m in self.memory[-10:]] or [0.0])
        return f"""
        🌐 SIN v{self.VERSION}
        Узлов: {len(self.nodes)}
        Память: {len(self.memory)}
        Семантических эпизодов: {len(self.semantic_memory.episodes)}
        Состояние: {'Спит' if self.sleeping else 'Бодрствует'}
        Вопросов: {len(self.pending_questions)}
        Целей: {len([g for g in self.goals if not g.achieved])}
        Эмоции: {[f'{k}:{v.intensity:.2f}' for k,v in self.emotions.items()]}
        Среднее понимание: {understanding_avg:.2f}
        CPU: {psutil.cpu_percent():.1f}%, RAM: {psutil.virtual_memory().percent:.1f}%
        """

    def show_memory(self, k=5):
        top = sorted(self.memory, key=lambda x: x.timestamp, reverse=True)[:k]
        print("\n🧠 Последние воспоминания:")
        for m in top:
            print(f"  [{m.level}] '{m.text}'")

    def show_episodes(self, k=5):
        recent = self.semantic_memory.episodes[-k:]
        print(f"\n🎬 Последние {k} эпизодов:")
        for ep in recent:
            when = time.strftime("%H:%M", time.localtime(ep.timestamp))
            print(f"  [{when}] '{ep.text}'")

    def visualize_resonance(self):
        if not self.activation_history:
            print("Нет данных для визуализации.")
            return
        plt.figure(figsize=(10, 5))
        data = np.array(self.activation_history)
        plt.imshow(data.T, aspect='auto', cmap='plasma', interpolation='none')
        plt.colorbar(label="Активация")
        plt.title("Волны резонанса в Sin")
        plt.xlabel("Время")
        plt.ylabel("Нейроны")
        plt.tight_layout()
        plt.show()

    def visualize_clusters(self):
        self.semantic_memory.visualize_clusters()

    def clear_memory(self):
        self.memory = []
        self.nodes = {}
        self.node_counter = 0
        self.phase_clusters = []
        self.vector_index = VectorIndex(dim=self.embedder.dim)
        self.semantic_memory.episodes = []
        self.semantic_memory.collection = self.semantic_memory.chroma_client.get_or_create_collection(name="semantic_episodes")
        logger.info("🧠 Память полностью очищена.")
        print("🧠 Память очищена.")

    def add_goal(self, description: str, priority: float = 0.5):
        goal = Goal(description=description, priority=priority, steps=[])
        self.goals.append(goal)
        logger.info(f"🎯 Добавлена цель: {description}")
        return f"Цель добавлена: {description}"

    def show_goals(self):
        if not self.goals:
            return "Нет активных целей"
        result = "\n🎯 Цели:\n"
        for goal in self.goals:
            status = "✅" if goal.achieved else "⏳"
            result += f"  {status} {goal.description} (приоритет: {goal.priority:.2f})\n"
        return result

    def create_agent(self, name: str):
        return self.multi_agent_system.create_agent(name)

    def communicate_agents(self, sender: str, receiver: str, message: str):
        return self.multi_agent_system.communicate(sender, receiver, message)

    def show_knowledge_graph(self):
        nodes = list(self.knowledge_graph.graph.nodes())
        edges = list(self.knowledge_graph.graph.edges(data=True))
        result = f"📊 Граф знаний: {len(nodes)} понятий, {len(edges)} связей\n"
        sample_edges = edges[:10]
        for u, v, data in sample_edges:
            rel = data.get('relation', 'unknown')
            result += f"  {u} --({rel})--> {v}\n"
        if len(edges) > 10:
            result += f"  ... и ещё {len(edges) - 10} связей.\n"
        return result

# === API ===
app = FastAPI(title=f"SIN API v{Sin.VERSION}")
sin = Sin()

@app.post("/learn")
async def api_learn(data: dict):
    try:
        text = data.get("text", "")
        result = sin.learn(text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/respond")
async def api_respond(data: dict):
    try:
        text = data.get("text", "")
        response = sin.respond(text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def api_status():
    return {
        "version": Sin.VERSION,
        "time": sin.t,
        "nodes": len(sin.nodes),
        "memory": len(sin.memory),
        "sleeping": sin.sleeping,
        "cognitive_load": sin.cognitive_load,
        "questions": len(sin.pending_questions)
    }

@app.post("/autonomous_learn")
async def api_autonomous_learn(data: dict):
    duration = data.get("duration", 30)
    result = sin.start_autonomous_learning(duration)
    return {"result": result}

@app.post("/stop_learning")
async def api_stop_learning():
    sin.stop_autonomous_learning()
    return {"result": "Обучение остановлено"}

@app.get("/autonomous_learn_log")
async def api_autonomous_learn_log():
    log = sin.get_autonomous_learning_log()
    return {"log": log}

@app.post("/add_goal")
async def api_add_goal(data: dict):
    description = data.get("description", "")
    priority = data.get("priority", 0.5)
    result = sin.add_goal(description, priority)
    return {"result": result}

# === TELEGRAM-БОТ ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Привет! Я SIN v{Sin.VERSION}. Давай пообщаемся!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if user_text.lower().startswith("good") or user_text.lower().startswith("bad"):
        feedback = "good" if "good" in user_text.lower() else "bad"
        sin.learn("feedback", user_feedback=feedback)
        await update.message.reply_text(f"✅ Ответ оценён как '{feedback}'")
    else:
        response = sin.respond(user_text)
        await update.message.reply_text(f"💬 Sin: {response}")

def run_telegram():
    app_bot = Application.builder().token(TELEGRAM_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_bot.run_polling()

# === CLI ===
def run_cli():
    print(sin.status())
    print("""
🔧 Доступные команды:
  !status — статус
  !sleep — заставить поспать
  !visualize — график резонанса
  !clusters — визуализация кластеров
  !memory — показать память
  !episodes — показать эпизоды
  !search "запрос" — семантический поиск
  !stats — статистика CPU/RAM
  !clear — очистить память
  !feedback good/bad — оценить ответ
  !autolearn <минуты> — автономное обучение
  !stoplearn — остановить обучение
  !goal <текст> — добавить цель
  !goals — показать цели
  !graph — показать граф знаний
  !autolog — показать лог автономного обучения
  !quit — выход
""")
    while True:
        try:
            user_input = input("> Sin, ").strip()
            if user_input.lower() == "!quit":
                break
            elif user_input.lower() == "!status":
                print(sin.status())
            elif user_input.lower() == "!sleep":
                sin.start_sleep()
            elif user_input.lower() == "!visualize":
                sin.visualize_resonance()
            elif user_input.lower() == "!clusters":
                sin.visualize_clusters()
            elif user_input.lower() == "!memory":
                sin.show_memory()
            elif user_input.lower() == "!episodes":
                sin.show_episodes()
            elif user_input.startswith("!search"):
                query = user_input[8:].strip().strip('"')
                if query:
                    result = sin.semantic_search(query)
                    print(f"🔍 Результат: {result}")
            elif user_input.lower() == "!stats":
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                print(f"\n📊 Статистика:\n  CPU: {cpu:.1f}%\n  RAM: {memory:.1f}%")
            elif user_input.lower() == "!clear":
                sin.clear_memory()
            elif user_input.startswith("!feedback"):
                fb = user_input.split()[-1]
                if fb in ["good", "bad"]:
                    sin.learn("feedback", user_feedback=fb)
                    print(f"✅ Ответ оценён как '{fb}'")
                else:
                    print("Используй: !feedback good или !feedback bad")
            elif user_input.startswith("!autolearn"):
                parts = user_input.split()
                duration = int(parts[1]) if len(parts) > 1 else 30
                result = sin.start_autonomous_learning(duration)
                print(f"🤖 {result}")
            elif user_input.lower() == "!stoplearn":
                sin.stop_autonomous_learning()
                print("🛑 Обучение остановлено")
            elif user_input.startswith("!goal"):
                goal_text = user_input[5:].strip()
                if goal_text:
                    result = sin.add_goal(goal_text)
                    print(f"🎯 {result}")
            elif user_input.lower() == "!goals":
                print(sin.show_goals())
            elif user_input.lower() == "!graph":
                print(sin.show_knowledge_graph())
            elif user_input.lower() == "!autolog":
                log = sin.get_autonomous_learning_log()
                if log:
                    print("\n📝 Лог автономного обучения:")
                    for entry in log:
                        print(f"  {entry}")
                else:
                    print("📝 Лог автономного обучения пуст.")
            else:
                learn_result = sin.learn(user_input)
                print(learn_result["response"])
                response = sin.respond(user_input)
                print(f"💬 Sin: {response}")
        except KeyboardInterrupt:
            break
    sin.save_state()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "telegram", "cli"], default="cli")
    args = parser.parse_args()
    if args.mode == "api":
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif args.mode == "telegram":
        run_telegram()
    else:
        run_cli()
