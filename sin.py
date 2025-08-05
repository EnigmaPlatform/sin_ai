import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
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
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import requests
import gzip
import shutil
from tqdm import tqdm
import psutil

# === НАСТРОЙКИ ===
EMBEDDING_PATH = r"C:\Users\alex\Downloads\cc.ru.300.vec"
EMBEDDING_GZ_PATH = EMBEDDING_PATH + ".gz"
EMBEDDING_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
BIN_PATH = "cc.ru.300.bin"
PERSIST_FILE = "sin_state.pkl"
LOG_FILE = "sin.log"
TELEGRAM_TOKEN = "7990254673:AAE-7UGlXLWnQ-Dn5D2uyrz0RYDJnBZZKM8"
MAX_NODES = 5000
SLEEP_CYCLE = 15
FORGET_THRESHOLD = 0.1
ATTENTION_DECAY = 0.93
GENERATION_TEMP = 0.7
DISSONANCE_THRESHOLD = 0.4
SAVE_INTERVAL = 300
MEMORY_HISTORY_LIMIT = 1000
MAX_CONTEXT_LENGTH = 10
MAX_TEXT_LENGTH = 500
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


# === RuEmbedder с кэшированием и прогрессом ===
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
        word_clean = word.lower().strip(".,!?\"'()[]{}:;—-")
        if not word_clean:
            return np.zeros(self.dim)
        if word_clean in self.model:
            return self.model[word_clean].copy()
        try:
            similar = self.model.most_similar(positive=[word_clean], topn=1)
            logger.debug(f"⚠️ '{word}' → '{similar[0][0]}'")
            return self.model[similar[0][0]].copy()
        except:
            logger.debug(f"⚠️ '{word}' неизвестно")
            return np.random.normal(0, 0.1, self.dim)

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
    vector: List[float]  # храним как list для pickle
    text: str
    level: int
    timestamp: float
    access_count: int = 0
    phase_cluster_id: Optional[int] = None
    reward_score: float = 0.0

    @staticmethod
    def from_dict(data):
        return MemoryItem(**data)

    def to_dict(self):
        return {
            'vector': self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            'text': self.text,
            'level': self.level,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'phase_cluster_id': self.phase_cluster_id,
            'reward_score': self.reward_score
        }

    @staticmethod
    def from_np_vector(vec: np.ndarray, text: str, level: int, timestamp: float, **kwargs):
        return MemoryItem(
            vector=vec.tolist(),
            text=text,
            level=level,
            timestamp=timestamp,
            **kwargs
        )


@dataclass
class DialogContext:
    last_messages: List[str] = None
    current_theme: Optional[str] = None
    thematic_attention: float = 0.0

    def __post_init__(self):
        if self.last_messages is None:
            self.last_messages = []


# === РЕЗОНАТОР ===
class Resonator:
    __slots__ = ['id', 'freq', 'phase', 'amplitude', 'damping', 'connections',
                 'pattern', 'level', 'last_activation', 'attention', 'phase_history']

    def __init__(self, node_id: int, freq: float = 1.0, phase: float = 0.0,
                 damping: float = 0.1, level: int = 0):
        self.id = node_id
        self.freq = freq
        self.phase = phase
        self.amplitude = 0.0
        self.damping = damping
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
        self.phase_history.append(self.phase)

    def step(self, dt: float = 0.1):
        if self.amplitude > 0.01:
            self.phase += self.freq * dt
            self.phase %= (2 * np.pi)
            self.amplitude *= (1 - self.damping * dt)
            self.attention *= ATTENTION_DECAY
            self.phase_history.append(self.phase)
        else:
            self.amplitude = 0.0


# === SIN — СЕТЬ ИНТУИТИВНОГО ПОНИМАНИЯ ===
class Sin:
    VERSION = "10.0"

    def __init__(self, persist_file: str = PERSIST_FILE):
        self.embedder = RuEmbedder()
        self.nodes = {}
        self.node_counter = 0
        self.memory = []
        self.activation_history = deque(maxlen=100)
        self.t = 0
        self.sleeping = False
        self.hierarchy_levels = 3
        self.level_nodes = [[] for _ in range(self.hierarchy_levels)]
        self.word_frequency = defaultdict(int)
        self.cognitive_load = 0.0
        self.pending_questions = []
        self.dialog_context = DialogContext()
        self.persist_file = persist_file
        self.last_save_time = time.time()
        self.cluster_labels = []
        self.rl_policy = {"ask_question": 0.7, "generate": 0.5}
        self.phase_clusters = []  # ✅ ИСПРАВЛЕНО: теперь инициализирован
        self._init_system()
        logger.info(f"Система SIN v{self.VERSION} инициализирована")

    def _init_system(self):
        if os.path.exists(self.persist_file):
            self._load_state()
        else:
            logger.info("Создана новая модель")

    def _load_state(self):
        try:
            with open(self.persist_file, 'rb') as f:
                data = pickle.load(f)
            # Восстанавливаем все поля
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            # Конвертируем вектора в np.ndarray
            self.memory = [
                MemoryItem(
                    vector=np.array(item['vector']),
                    text=item['text'],
                    level=item['level'],
                    timestamp=item['timestamp'],
                    access_count=item.get('access_count', 0),
                    phase_cluster_id=item.get('phase_cluster_id'),
                    reward_score=item.get('reward_score', 0.0)
                ) for item in self.memory
            ]
            # Восстанавливаем phase_clusters, если нет
            if not hasattr(self, 'phase_clusters'):
                self.phase_clusters = []
            logger.info(f"Состояние загружено из {self.persist_file}")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}")

    def save_state(self):
        try:
            serializable_memory = [item.to_dict() for item in self.memory]
            serializable_word_freq = dict(self.word_frequency)
            serializable_nodes = {
                k: {
                    'id': v.id,
                    'freq': v.freq,
                    'phase': v.phase,
                    'amplitude': v.amplitude,
                    'damping': v.damping,
                    'connections': dict(v.connections),
                    'pattern': v.pattern.tolist() if isinstance(v.pattern, np.ndarray) else v.pattern,
                    'level': v.level,
                    'last_activation': v.last_activation,
                    'attention': v.attention,
                    'phase_history': list(v.phase_history)
                } for k, v in self.nodes.items()
            }
            with open(self.persist_file, 'wb') as f:
                pickle.dump({
                    'nodes': serializable_nodes,
                    'node_counter': self.node_counter,
                    'memory': serializable_memory,
                    't': self.t,
                    'word_frequency': serializable_word_freq,
                    'level_nodes': self.level_nodes,
                    'cluster_labels': self.cluster_labels,
                    'rl_policy': self.rl_policy,
                    'phase_clusters': self.phase_clusters,
                    'dialog_context': {
                        'last_messages': self.dialog_context.last_messages,
                        'current_theme': self.dialog_context.current_theme,
                        'thematic_attention': self.dialog_context.thematic_attention
                    }
                }, f)
            logger.info(f"Состояние сохранено в {self.persist_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {str(e)}")

    def _auto_save(self):
        if time.time() - self.last_save_time > SAVE_INTERVAL:
            self.save_state()
            self.last_save_time = time.time()

    def tokenize(self, text: str) -> List[str]:
        words = [word.strip(".,!?\"'()[]{}:;—-") for word in text.lower().split() if word.isalpha()]
        for word in words:
            self.word_frequency[word] += 1
        return words

    def _update_dialog_context(self, text: str):
        self.dialog_context.last_messages.append(text)
        if len(self.dialog_context.last_messages) >= 3:
            recent_text = " ".join(self.dialog_context.last_messages)
            words = self.tokenize(recent_text)
            if not words:
                return
            theme_vector = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
            if self.dialog_context.current_theme is None:
                self.dialog_context.current_theme = hashlib.md5(theme_vector.tobytes()).hexdigest()
                self.dialog_context.thematic_attention = 0.5
            else:
                old_theme_vec = self.embedder.get_vector(self.dialog_context.current_theme[:10])
                similarity = cosine_similarity([theme_vector], [old_theme_vec])[0][0]
                self.dialog_context.thematic_attention = 0.3 * self.dialog_context.thematic_attention + 0.7 * similarity

    def are_in_phase(self, node1: Resonator, node2: Resonator, tol=0.5) -> bool:
        return abs((node1.phase - node2.phase) % (2 * np.pi)) < tol

    def assign_to_phase_cluster(self, node: Resonator) -> int:
        for cluster_id, cluster in enumerate(self.phase_clusters):
            if cluster:
                first_node_id = cluster[0]
                if first_node_id in self.nodes:
                    if self.are_in_phase(node, self.nodes[first_node_id]):
                        cluster.append(node.id)
                        return cluster_id
        new_cluster = [node.id]
        self.phase_clusters.append(new_cluster)
        return len(self.phase_clusters) - 1

    def hierarchical_forget(self):
        to_remove = []
        for i, mem in enumerate(self.memory):
            if isinstance(mem.text, str):
                words = self.tokenize(mem.text)
                freq_score = sum(self.word_frequency.get(w, 0) for w in words) / (len(words) + 1e-8)
                forget_bias = 0.5 if mem.level == 0 else 0.1
                if freq_score < FORGET_THRESHOLD * forget_bias:
                    to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.memory.pop(i)
        if to_remove:
            logger.info(f"🧹 Иерархически забыто {len(to_remove)} элементов")

    def generate_question(self, word: str) -> str:
        try:
            similar = self.embedder.model.most_similar(positive=[word], topn=1)
            return f"Я не до конца понимаю '{word}'. Это похоже на '{similar[0][0]}'? Или чем отличается?"
        except:
            return f"Что такое '{word}'? Можешь объяснить проще?"

    def learn(self, text: str, from_dialog: bool = False, user_feedback: str = "neutral") -> Dict:
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        self._update_dialog_context(text)
        words = self.tokenize(text)
        if not words:
            return {"status": "empty", "response": "Пустой ввод"}

        total_vec = np.zeros(self.embedder.dim)

        for word in words:
            vec = self.embedder.get_vector(word)
            total_vec += vec

            new_id = self.node_counter
            node = Resonator(new_id, level=0)
            node.pattern = vec.copy()
            self.nodes[new_id] = node
            node.excite(1.0)
            self.node_counter += 1

            resonance = self.check_resonance(vec)
            if resonance < 0.6:
                cluster_id = self.assign_to_phase_cluster(node)
                self.memory.append(MemoryItem.from_np_vector(
                    vec, word, level=0, timestamp=time.time(), phase_cluster_id=cluster_id
                ))

            if resonance < DISSONANCE_THRESHOLD and not from_dialog and random.random() < 0.7:
                question = self.generate_question(word)
                self.pending_questions.append(question)

        total_vec /= len(words)
        self.memory.append(MemoryItem.from_np_vector(
            total_vec, ' '.join(words), level=1, timestamp=time.time()
        ))

        self.hierarchical_forget()
        self._auto_save()

        # RL: обучение с подкреплением
        if user_feedback == "good":
            self.rl_policy["ask_question"] = min(1.0, self.rl_policy["ask_question"] + 0.1)
            self.rl_policy["generate"] = min(1.0, self.rl_policy["generate"] + 0.1)
        elif user_feedback == "bad":
            self.rl_policy["ask_question"] = max(0.0, self.rl_policy["ask_question"] - 0.1)
            self.rl_policy["generate"] = max(0.0, self.rl_policy["generate"] - 0.1)

        return {"status": "learned", "response": f"Sin понял: '{text}'"}

    def respond(self, text: str) -> str:
        if self.pending_questions and random.random() < self.rl_policy["ask_question"]:
            return f"❓ {self.pending_questions.pop(0)}"
        if self.sleeping:
            return "Zzz... Sin спит."
        words = self.tokenize(text)
        if not words:
            return "Я слушаю..."
        query_vec = np.mean([self.embedder.get_vector(w) for w in words], axis=0)
        best_sim = 0.0
        best_match = None
        for mem in self.memory:
            sim = cosine_similarity([query_vec], [mem.vector])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_match = mem.text
        if best_sim > 0.6:
            hints = ["Это напоминает мне о", "Я чувствую сходство с"]
            return f"{random.choice(hints)} '{best_match}' (схожесть: {best_sim:.2f})."
        elif best_sim > 0.4:
            return f"Частично понимаю. Ещё не до конца ясно ({best_sim:.2f})."
        else:
            return f"Новое. Ещё не резонирует. Расскажи больше."

    def check_resonance(self, vec) -> float:
        sims = []
        for mem in self.memory:
            sim = cosine_similarity([vec], [mem.vector])[0][0]
            if sim > 0.2:
                sims.append(sim)
        return max(sims) if sims else 0.0

    def generate_response(self, seed: str, length=5) -> str:
        base_sequence = self.embedder.generate_sequence(seed, length=length)
        enhanced = []
        for word in base_sequence:
            try:
                sims = [
                    (cosine_similarity([self.embedder.get_vector(word)], [m.vector])[0][0], m.text)
                    for m in self.memory if isinstance(m.text, str)
                ]
                if sims:
                    best_sim, best_text = max(sims)
                    if best_sim > 0.6:
                        enhanced.append(best_text)
                    else:
                        enhanced.append(word)
                else:
                    enhanced.append(word)
            except:
                enhanced.append(word)
        return " ".join(enhanced[:length])

    def status(self):
        return f"""
        🌐 Sin v{self.VERSION} — Сеть Интуитивного Понимания
        Время: {self.t}
        Узлов: {len(self.nodes)}
        Память: {len(self.memory)}
        Состояние: {'Спит' if self.sleeping else 'Бодрствует'}
        Уровни: {len(self.level_nodes[0])} слов, {len(self.level_nodes[1])} фраз
        Нагрузка: {self.cognitive_load:.2f}
        Вопросов: {len(self.pending_questions)}
        """

    def start_sleep(self):
        self.sleeping = True
        logger.info(f"🌙 Sin засыпает... (нагрузка: {self.cognitive_load:.2f})")
        threading.Thread(target=self.dream_cycle, daemon=True).start()

    def dream_cycle(self):
        time.sleep(1)
        logger.info("🧠 Sin видит сны...")
        for _ in range(5):
            if len(self.memory) == 0:
                continue
            mem = random.choice(self.memory)
            vec = mem.vector
            noise = np.random.normal(0, 0.05, vec.shape)
            dream = vec + noise
            dream /= (np.linalg.norm(dream) + 1e-8)
            if self.check_resonance(dream) < 0.8:
                self.memory.append(MemoryItem.from_np_vector(
                    dream, f"[сон:{mem.text}]", level=mem.level, timestamp=self.t
                ))
            time.sleep(0.5)
        self.sleeping = False
        self.cognitive_load *= 0.5
        logger.info("✨ Sin проснулся. Память укреплена.")

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

    def show_memory(self, top_k=10):
        sorted_mem = sorted(self.memory, key=lambda x: x.timestamp, reverse=True)
        print(f"\n🧠 Последние {top_k} воспоминаний:")
        for mem in sorted_mem[:top_k]:
            print(f"  [{mem.level}] '{mem.text}' ({mem.timestamp:.0f})")

    def show_questions(self):
        if not self.pending_questions:
            print("Нет открытых вопросов.")
        else:
            print(f"\n❓ {len(self.pending_questions)} вопросов:")
            for q in self.pending_questions:
                print(f"  • {q}")

    def show_stats(self):
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        print(f"\n📊 Статистика:")
        print(f"  CPU: {cpu:.1f}%")
        print(f"  RAM: {memory:.1f}%")
        print(f"  Количество слов: {len(self.word_frequency)}")
        print(f"  Общий объём памяти: {len(self.memory)}")

    def clear_memory(self):
        self.memory = []
        self.nodes = {}
        self.node_counter = 0
        self.phase_clusters = []
        logger.info("🧠 Память полностью очищена.")
        print("🧠 Память очищена.")


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

@app.post("/generate")
async def api_generate(data: dict):
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
  !questions — показать вопросы
  !stats — статистика CPU/RAM
  !clear — очистить память
  !generate <слово> — сгенерировать текст
  !feedback good/bad — оценить ответ
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
            elif user_input.lower() == "!questions":
                sin.show_questions()
            elif user_input.lower() == "!stats":
                sin.show_stats()
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
            else:
                learn_result = sin.learn(user_input)
                print(learn_result["response"])
                response = sin.respond(user_input)
                print(f"💬 Sin: {response}")
        except KeyboardInterrupt:
            break


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
