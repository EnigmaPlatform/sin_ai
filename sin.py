import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
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
from typing import List, Dict, Optional, Tuple, Any

# === НАСТРОЙКИ ПУТЕЙ ===
BASE_PATH = r"C:\Users\alex\Downloads"
EMBEDDING_PATH = os.path.join(BASE_PATH, "cc.ru.300.vec")
EMBEDDING_GZ_PATH = EMBEDDING_PATH + ".gz"
EMBEDDING_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
BIN_PATH = os.path.join(BASE_PATH, "cc.ru.300.bin")
PERSIST_FILE = os.path.join(BASE_PATH, "sin_state.pkl")
GRAPH_FILE = os.path.join(BASE_PATH, "knowledge_graph.pkl")
LOG_FILE = os.path.join(BASE_PATH, "sin.log")
TELEGRAM_TOKEN = "7990254673:AAE-7UGlXLWnQ-Dn5D2uyrz0RYDJnBZZKM8"

# === ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ===
MAX_NODES = 10000
SLEEP_CYCLE = 15
SAVE_INTERVAL = 1800  # автосохранение каждые 30 минут
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

# === ФУНКЦИИ ДЛЯ СКАЧИВАНИЯ И РАСПАКОВКИ ===
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

# === RuEmbedder с кэшированием, прогрессом и нормализацией ===
class RuEmbedder:
    def __init__(self, filepath=EMBEDDING_PATH):
        if not check_and_download_embeddings():
            raise RuntimeError("Не удалось подготовить эмбеддинги.")
        if os.path.exists(BIN_PATH):
            logger.info("🌀 Загрузка из бинарного файла (быстро)...")
            self.model = KeyedVectors.load(BIN_PATH)
        else:
            logger.info("🌀 Загрузка из текстового файла (ограничено 50k слов)...")
            total_lines = 50000 + 1
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

    def generate_sequence(self, seed_word: str, length=5, diversity=1.0) -> List[str]:
        sequence = [seed_word]
        current_word = seed_word
        for _ in range(length - 1):
            try:
                similar = self.model.most_similar(positive=[current_word], topn=10)
                words, scores = zip(*similar)
                probs = np.array(scores) ** (1 / diversity)
                probs /= probs.sum()
                next_word = np.random.choice(words, p=probs)
                sequence.append(next_word)
                current_word = next_word
            except:
                break
        return sequence

# === СТРУКТУРЫ ДАННЫХ ===
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

# === Resonator (нейрон с фазой) ===
class Resonator:
    __slots__ = ['id', 'freq', 'phase', 'amplitude', 'damping', 'connections',
                 'pattern', 'level', 'last_activation', 'attention', 'phase_history']
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
        else:
            self.amplitude = 0.0

# === Hippocampus — кратковременная память и консолидация ===
class Hippocampus:
    def __init__(self, capacity: int = WORKING_MEMORY_SIZE):
        self.working_memory = deque(maxlen=capacity)
        self.consolidation_threshold = 0.7

    def add(self, item: MemoryItem):
        self.working_memory.append(item)

    def consolidate(self, long_term_memory: list, vector_index):
        for item in self.working_memory:
            if item.coherence_score > self.consolidation_threshold:
                long_term_memory.append(item)
                vector_index.add_vector(item.to_np_vector(), len(long_term_memory) - 1)
        self.working_memory.clear()

# === VectorIndex — быстрый поиск (FAISS) ===
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

# === AdaptiveParams — адаптивные пороги и обучение ===
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

# === KnowledgeGraph — граф знаний ===
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concepts = {}  # concept_name -> vector

    def add_concept(self, name: str, vector: np.ndarray, parents: List[str] = None, children: List[str] = None):
        if parents is None:
            parents = []
        if children is None:
            children = []
            
        self.graph.add_node(name)
        self.concepts[name] = vector
        
        for parent in parents:
            self.graph.add_edge(parent, name, relation="hypernym")
        for child in children:
            self.graph.add_edge(name, child, relation="hyponym")
            
    def find_path(self, source: str, target: str) -> List[str]:
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    def get_neighbors(self, concept: str, depth=1) -> List[str]:
        if not self.graph.has_node(concept):
            return []
        neighbors = list(self.graph.neighbors(concept))
        if depth > 1:
            for neighbor in list(neighbors):
                neighbors.extend(self.get_neighbors(neighbor, depth-1))
        return list(set(neighbors))

    def save_graph(self):
        try:
            with open(GRAPH_FILE, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'concepts': self.concepts
                }, f)
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

# === AutonomousLearner — автономное обучение во сне ===
class AutonomousLearner:
    def __init__(self, sin_instance):
        self.sin = sin_instance
        self.is_learning = False
        self.learning_thread = None
        
    def start_autonomous_learning(self, duration_minutes: int = 30):
        if self.is_learning:
            return "Обучение уже запущено"
            
        self.is_learning = True
        logger.info(f"🤖 Запуск автономного обучения на {duration_minutes} минут")
        
        def learning_loop():
            end_time = time.time() + duration_minutes * 60
            cycle = 0
            
            while time.time() < end_time and self.is_learning:
                cycle += 1
                logger.info(f"🔄 Автономный цикл {cycle}")
                
                # 1. Генерация гипотез
                self.generate_hypotheses()
                
                # 2. Укрепление связей
                self.strengthen_connections()
                
                # 3. Обнаружение конфликтов
                self.detect_conflicts()
                
                # 4. Автосохранение каждые 30 минут
                if cycle % 6 == 0:  # каждые 30 минут при 5-минутных циклах
                    self.sin.save_state()
                    self.sin.knowledge_graph.save_graph()
                    
                time.sleep(300)  # 5 минут на цикл
                
            self.is_learning = False
            logger.info("✅ Автономное обучение завершено")
            
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()
        return f"Обучение запущено на {duration_minutes} минут"
        
    def stop_learning(self):
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=1)
        logger.info("🛑 Автономное обучение остановлено")
        
    def generate_hypotheses(self):
        if len(self.sin.memory) < 2:
            return
            
        # Выбираем два случайных элемента памяти
        mem1, mem2 = random.sample(self.sin.memory, 2)
        vec1, vec2 = mem1.to_np_vector(), mem2.to_np_vector()
        
        # Векторная арифметика: a - b + c = ?
        hypothesis_vec = vec1 - vec2 + np.random.normal(0, 0.05, vec1.shape)
        hypothesis_vec /= (np.linalg.norm(hypothesis_vec) + 1e-8)
        
        # Поиск ближайшего слова
        results = self.sin.vector_index.search_similar(hypothesis_vec, k=1)
        if results and results[0][0] > 0.5:
            target_text = self.sin.memory[results[0][1]].text
            question = f"Я заметил связь между '{mem1.text}' и '{mem2.text}'. Возможно, '{target_text}' — это результат этой связи?"
            self.sin.pending_questions.append(question)
            
    def strengthen_connections(self):
        # Увеличиваем коэренцию часто используемых связей
        for mem in self.sin.memory:
            if mem.access_count > 5:
                mem.coherence_score = min(1.0, mem.coherence_score + 0.1)
                
    def detect_conflicts(self):
        # Поиск противоречивых утверждений
        for i, mem1 in enumerate(self.sin.memory):
            for j, mem2 in enumerate(self.sin.memory[i+1:], i+1):
                sim = cosine_similarity([mem1.to_np_vector()], [mem2.to_np_vector()])[0][0]
                if sim > 0.8 and ("не " in mem1.text) != ("не " in mem2.text):
                    conflict = f"Обнаружено противоречие: '{mem1.text}' vs '{mem2.text}' (схожесть: {sim:.2f})"
                    logger.warning(conflict)
                    self.sin.pending_questions.append(f"Я нашёл противоречие: {mem1.text} и {mem2.text}. Какое утверждение верно?")

# === MultiAgentSystem — система мультиагентов ===
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
        
        # Отправитель учится
        sender_agent.learn(message)
        
        # Получатель отвечает
        response = receiver_agent.respond(message)
        
        self.communication_history.append({
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "response": response,
            "timestamp": time.time()
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

# === SIN — ОБЪЕДИНЁННАЯ СИСТЕМА ===
class Sin:
    VERSION = "17.0"
    def __init__(self, persist_file: str = PERSIST_FILE):
        self.embedder = RuEmbedder()
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
            sim = cosine_similarity([query_vec], [self.embedder.get_vector(msg.split()[-1])])[0][0]
            context_relevance += sim * decay
        context_relevance /= max(len(self.dialog_context.last_messages), 1)
        return 0.6 * top_sim + 0.4 * context_relevance

    def learn(self, text: str, user_feedback: str = "neutral") -> Dict:
        understanding = self.calculate_understanding_score(text)
        words = self.tokenize(text)
        if not words:
            return {"status": "empty", "response": "Пусто", "understanding": understanding}

        total_vec = np.zeros(self.embedder.dim)
        reward = 0.0

        for word in words:
            vec = self.embedder.get_vector(word)
            total_vec += vec
            conflicts = self.conflict_detector(vec)
            if conflicts:
                reward -= 0.5
            nid = self.node_counter
            node = Resonator(nid)
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
            self.hippocampus.add(mem_item)
            
            # Добавляем в граф знаний
            if len(conflicts) == 0:
                self.knowledge_graph.add_concept(word, vec)

        total_vec /= len(words)
        phrase_item = MemoryItem.from_np(
            vector=total_vec,
            text=' '.join(words),
            level=1,
            timestamp=time.time(),
            reward_score=reward
        )
        self.hippocampus.add(phrase_item)
        self.hippocampus.consolidate(self.memory, self.vector_index)

        if user_feedback == "good":
            reward = RL_REWARD_CORRECT
            self.emotions["certainty"].intensity = min(1.0, self.emotions["certainty"].intensity + 0.1)
        elif user_feedback == "bad":
            reward = RL_PENALTY_WRONG
            self.emotions["certainty"].intensity = max(0.0, self.emotions["certainty"].intensity - 0.2)
            
        self.emotions["curiosity"].intensity = min(1.0, self.emotions["curiosity"].intensity + 0.05)
        self.params.update(reward)
        self.rl_policy["ask_question"] = 0.5 + 0.5 * (reward / 2.0) if reward != 0 else 0.7

        self._auto_save()

        status = "understood" if understanding > UNDERSTANDING_THRESHOLD else "partially"
        return {
            "status": status,
            "response": f"{'🧠' if understanding > 0.7 else '🤔'} Понял: '{text}' (понимание: {understanding:.2f})",
            "understanding": understanding
        }

    def respond(self, text: str) -> str:
        if self.sleeping:
            return "Zzz... Sin спит."
        understanding = self.calculate_understanding_score(text)
        if understanding < 0.3:
            return "❓ Совсем новое. Расскажи подробнее."

        words = self.tokenize(text)
        if not words:
            return "Я слушаю..."
        query_vec = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
        results = self.vector_index.search_similar(query_vec, k=10)

        # Проверка эмоций для определения поведения
        curiosity = self.emotions["curiosity"].intensity
        if self.pending_questions and random.random() < self.rl_policy["ask_question"] * curiosity:
            return f"❓ {self.pending_questions.pop(0)}"

        if results and results[0][0] > 0.6:
            best_text = self.memory[results[0][1]].text
            return f"🧠 Это напоминает: '{best_text}' (схожесть: {results[0][0]:.2f})"
        return f"🤔 Частично понимаю. Ещё не до конца ясно."

    def generate_response(self, seed: str, length=5) -> str:
        base_sequence = self.embedder.generate_sequence(seed, length=length)
        enhanced = []
        for word in base_sequence:
            vec = self.embedder.get_vector(word)
            results = self.vector_index.search_similar(vec, k=5)
            if results and results[0][0] > 0.5:
                enhanced.append(self.memory[results[0][1]].text)
            else:
                enhanced.append(word)
        return " ".join(enhanced[:length])

    def dream_cycle(self):
        logger.info("💭 Sin видит сны...")
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
            time.sleep(0.3)
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

    def status(self):
        understanding_avg = np.mean([self.calculate_understanding_score(m.text) for m in self.memory[-10:]] or [0.0])
        return f"""
        🌐 SIN v{self.VERSION}
        Узлов: {len(self.nodes)}
        Память: {len(self.memory)}
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

    def clear_memory(self):
        self.memory = []
        self.nodes = {}
        self.node_counter = 0
        self.phase_clusters = []
        self.vector_index = VectorIndex(dim=self.embedder.dim)
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
        edges = list(self.knowledge_graph.graph.edges())
        return f"📊 Граф знаний: {len(nodes)} понятий, {len(edges)} связей"

# === API ===
app = FastAPI(title=f"SIN API v{Sin.VERSION}")
sin = Sin()

@app.post("/learn")
async def api_learn( dict):
    try:
        text = data.get("text", "")
        result = sin.learn(text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/respond")
async def api_respond( dict):
    try:
        text = data.get("text", "")
        response = sin.respond(text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def api_generate( dict):
    try:
        seed = data.get("seed", "мысль")
        length = data.get("length", 5)
        text = sin.generate_response(seed, length)
        return {"generated": text}
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
    response = sin.respond(user_text)
    await update.message.reply_text(f"💬 Sin: {response}")

def run_telegram():
    app_bot = Application.builder().token(TELEGRAM_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_bot.run_polling()

# === КОНСОЛЬНЫЙ ИНТЕРФЕЙС ===
def run_cli():
    print(sin.status())
    print("""
🔧 Доступные команды:
  !status — статус
  !sleep — заставить поспать
  !visualize — график резонанса
  !memory — показать память
  !stats — статистика CPU/RAM
  !clear — очистить память
  !generate <слово> — сгенерировать текст
  !feedback good/bad — оценить ответ
  !autolearn <минуты> — автономное обучение
  !stoplearn — остановить обучение
  !goal <текст> — добавить цель
  !goals — показать цели
  !graph — показать граф знаний
  !agent <имя> — создать агента
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
            elif user_input.lower() == "!memory":
                sin.show_memory()
            elif user_input.lower() == "!stats":
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                print(f"\n📊 Статистика:\n  CPU: {cpu:.1f}%\n  RAM: {memory:.1f}%")
            elif user_input.lower() == "!clear":
                sin.clear_memory()
            elif user_input.startswith("!generate"):
                parts = user_input.split()
                seed = parts[1] if len(parts) > 1 else "мысль"
                length = int(parts[2]) if len(parts) > 2 else 5
                generated = sin.generate_response(seed, length)
                print(f"🔮 Сгенерировано: '{generated}'")
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
            elif user_input.startswith("!agent"):
                agent_name = user_input[6:].strip()
                if agent_name:
                    sin.create_agent(agent_name)
                    print(f"🤖 Агент '{agent_name}' создан")
            else:
                learn_result = sin.learn(user_input)
                print(learn_result["response"])
                response = sin.respond(user_input)
                print(f"💬 Sin: {response}")
        except KeyboardInterrupt:
            break
    sin.save_state()

# === ЗАПУСК ===
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
