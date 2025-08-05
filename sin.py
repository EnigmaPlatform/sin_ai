import asyncio
import aiofiles
import aiohttp
import numpy as np
import hashlib
import logging
import pickle
import os
import time
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Dict, Optional
from gensim.models import KeyedVectors
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from contextlib import asynccontextmanager
import threading
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# === КОНФИГУРАЦИЯ ===
class Config:
    EMBEDDING_PATH = r"C:\Users\alex\Downloads\cc.ru.300.vec"
    EMBEDDING_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
    PERSIST_FILE = "sin_state.pkl"
    LOG_FILE = "sin.log"
    TELEGRAM_TOKEN = "7990254673:AAE-7UGlXLWnQ-Dn5D2uyrz0RYDJnBZZKM8"
    MAX_NODES = 5000
    SLEEP_CYCLE = 15
    FORGET_THRESHOLD = 0.1
    ATTENTION_DECAY = 0.93
    DISSONANCE_THRESHOLD = 0.4
    SAVE_INTERVAL = 300
    MAX_CONTEXT_LENGTH = 10
    MAX_HISTORY_LENGTH = 100
    MAX_TEXT_LENGTH = 500
    MAX_CONCURRENT_REQUESTS = 10
    MAX_MEMORY_PERCENT = 80
    CACHE_SIZE = 1000
    RL_REWARD_CORRECT = 2.0
    RL_REWARD_QUESTION = 1.0
    RL_PENALTY_WRONG = -1.0


# === ЛОГГИРОВАНИЕ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIN")


# === Pydantic МОДЕЛИ (V2) ===
class LearnRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_TEXT_LENGTH)

    @field_validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class RespondRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_TEXT_LENGTH)


# === АСИНХРОННЫЙ ХЭШ ===
async def async_get_md5(filepath: str) -> Optional[str]:
    hash_md5 = hashlib.md5()
    try:
        async with aiofiles.open(filepath, "rb") as f:
            while chunk := await f.read(8192):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Ошибка при вычислении MD5: {e}")
        return None


# === АСИНХРОННАЯ ЗАГРУЗКА ===
async def async_download_embeddings(session: aiohttp.ClientSession, url: str, path: str) -> bool:
    gz_path = path + ".gz"
    logger.info(f"Начинаю загрузку: {url}")
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            async with aiofiles.open(gz_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % max(total_size // 10, 1) == 0:
                            logger.info(f"Загрузка: {percent:.1f}%")

        import gzip
        with gzip.open(gz_path, 'rb') as f_in:
            with open(path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(gz_path)
        logger.info(f"Файл сохранён: {path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при загрузке: {str(e)}")
        return False


# === КЭШ ВЕКТОРОВ ===
class VectorCache:
    def __init__(self, maxsize=Config.CACHE_SIZE):
        self.cache = {}
        self.maxsize = maxsize
        self.lock = asyncio.Lock()

    async def get(self, word: str) -> Optional[np.ndarray]:
        async with self.lock:
            return self.cache.get(word)

    async def set(self, word: str, vector: np.ndarray):
        async with self.lock:
            if len(self.cache) >= self.maxsize:
                del self.cache[next(iter(self.cache))]
            self.cache[word] = vector.copy()


# === ЯДРО СИСТЕМЫ ===
@dataclass
class MemoryItem:
    vector: np.ndarray
    text: str
    level: int
    timestamp: float
    access_count: int = 0
    phase_cluster_id: Optional[int] = None
    reward_score: float = 0.0  # Для RL


@dataclass
class DialogContext:
    last_messages: deque = field(default_factory=lambda: deque(maxlen=Config.MAX_CONTEXT_LENGTH))
    current_theme: Optional[str] = None
    thematic_attention: float = 0.0


class RuEmbedder:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.model = None
        self.dim = 300
        self.cache = VectorCache()

    async def _load_embeddings(self) -> bool:
        try:
            logger.info("🌀 Загрузка эмбеддингов...")
            self.model = KeyedVectors.load_word2vec_format(self.filepath, binary=False, limit=300000)
            logger.info(f"✅ Загружено {len(self.model.key_to_index)} слов (dim={self.dim})")
            return True
        except Exception as e:
            logger.critical(f"Не удалось загрузить эмбеддинги: {str(e)}")
            return False

    async def get_vector(self, word: str) -> np.ndarray:
        cached = await self.cache.get(word)
        if cached is not None:
            return cached

        word_clean = word.lower().strip(".,!?\"'()[]{}:;—-")
        if not word_clean:
            return np.zeros(self.dim)

        if word_clean in self.model:
            vec = self.model[word_clean].copy()
        else:
            try:
                similar = self.model.most_similar(positive=[word_clean], topn=1)
                logger.debug(f"Слово '{word}' заменено на '{similar[0][0]}'")
                vec = self.model[similar[0][0]].copy()
            except:
                logger.debug(f"Неизвестное слово: '{word}'")
                vec = np.random.normal(0, 0.1, self.dim)

        await self.cache.set(word, vec)
        return vec


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
            self.attention *= Config.ATTENTION_DECAY
            self.phase_history.append(self.phase)
        else:
            self.amplitude = 0.0


class Sin:
    VERSION = "9.0"
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_file: str = Config.PERSIST_FILE):
        if hasattr(self, 'initialized'):
            return
        self.persist_file = persist_file
        self.embedder = None
        self.nodes = {}
        self.node_counter = 0
        self.memory = []
        self.activation_history = deque(maxlen=Config.MAX_HISTORY_LENGTH)
        self.t = 0
        self.sleeping = False
        self.hierarchy_levels = 3
        self.level_nodes = [[] for _ in range(self.hierarchy_levels)]
        self.word_frequency = defaultdict(int)
        self.cognitive_load = 0.0
        self.pending_questions = []
        self.dialog_context = DialogContext()
        self.last_save_time = time.time()
        self.phase_clusters = []
        self.request_count = 0
        self.error_count = 0
        self.save_lock = asyncio.Lock()
        self.learn_lock = asyncio.Lock()
        self.cluster_labels = []
        self.rl_policy = {"ask_question": 0.7, "generate": 0.5}  # Инициализация политики
        self.initialized = False

    async def _init_system(self):
        async with self._lock:
            if self.initialized:
                return
            logger.info("Инициализация системы...")
            await self._check_and_download_embeddings()
            self.embedder = RuEmbedder(Config.EMBEDDING_PATH)
            if not await self.embedder._load_embeddings():
                raise RuntimeError("Не удалось загрузить эмбеддинги")
            if os.path.exists(self.persist_file):
                await self._load_state()
            else:
                logger.info("Создана новая модель")
            asyncio.create_task(self._background_save())
            asyncio.create_task(self._monitor_resources())
            self.initialized = True
            logger.info(f"SIN v{self.VERSION} инициализирован")

    async def _check_and_download_embeddings(self):
        if not os.path.exists(Config.EMBEDDING_PATH):
            logger.warning(f"Файл не найден: {Config.EMBEDDING_PATH}")
            print("Файл эмбеддингов отсутствует. Начать загрузку? (y/n): ", end="")
            if input().lower() == 'y':
                async with aiohttp.ClientSession() as session:
                    if await async_download_embeddings(session, Config.EMBEDDING_URL, Config.EMBEDDING_PATH):
                        logger.info("Загрузка завершена.")
                    else:
                        raise RuntimeError("Не удалось загрузить эмбеддинги.")
            else:
                raise RuntimeError("Загрузка отменена.")

    async def _load_state(self):
        try:
            async with aiofiles.open(self.persist_file, 'rb') as f:
                data = await f.read()
                state = pickle.loads(data)
                for key, value in state.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            logger.info(f"Состояние загружено из {self.persist_file}")
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {str(e)}")

    async def _save_state(self):
        try:
            async with self.save_lock:
                async with aiofiles.open(self.persist_file, 'wb') as f:
                    data = {
                        'nodes': self.nodes,
                        'node_counter': self.node_counter,
                        'memory': self.memory,
                        't': self.t,
                        'word_frequency': self.word_frequency,
                        'level_nodes': self.level_nodes,
                        'phase_clusters': self.phase_clusters,
                        'cluster_labels': self.cluster_labels,
                        'rl_policy': self.rl_policy
                    }
                    await f.write(pickle.dumps(data))
            logger.info(f"Состояние сохранено в {self.persist_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {str(e)}")

    async def _background_save(self):
        while True:
            await asyncio.sleep(Config.SAVE_INTERVAL)
            if self.initialized:
                await self._save_state()

    async def _monitor_resources(self):
        process = psutil.Process()
        while True:
            await asyncio.sleep(10)
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            self.cognitive_load = memory / 100.0
            logger.info(f"Мониторинг: CPU={cpu:.1f}%, RAM={memory:.1f}%")

    async def _update_dialog_context(self, text: str):
        self.dialog_context.last_messages.append(text)
        if len(self.dialog_context.last_messages) >= 3:
            recent_text = " ".join(self.dialog_context.last_messages)
            theme_vector = np.mean([await self.embedder.get_vector(w) for w in await self.tokenize(recent_text)], axis=0)
            if self.dialog_context.current_theme is None:
                self.dialog_context.current_theme = hashlib.md5(theme_vector.tobytes()).hexdigest()
                self.dialog_context.thematic_attention = 0.5
            else:
                old_theme_vec = await self.embedder.get_vector(self.dialog_context.current_theme[:10])
                similarity = cosine_similarity([theme_vector], [old_theme_vec])[0][0]
                self.dialog_context.thematic_attention = 0.3 * self.dialog_context.thematic_attention + 0.7 * similarity

    async def tokenize(self, text: str) -> List[str]:
        words = [word.strip(".,!?\"'()[]{}:;—-") for word in text.lower().split() if word.isalpha()]
        for word in words:
            self.word_frequency[word] += 1
        return words

    def are_in_phase(self, node1: Resonator, node2: Resonator, tol=0.5) -> bool:
        return abs((node1.phase - node2.phase) % (2 * np.pi)) < tol

    def assign_to_phase_cluster(self, node: Resonator) -> int:
        for cluster_id, cluster in enumerate(self.phase_clusters):
            if cluster and self.are_in_phase(node, self.nodes[cluster[0]]):
                cluster.append(node.id)
                return cluster_id
        new_cluster = [node.id]
        self.phase_clusters.append(new_cluster)
        return len(self.phase_clusters) - 1

    async def hierarchical_forget(self):
        to_remove = []
        for i, mem in enumerate(self.memory):
            if isinstance(mem.text, str):
                words = await self.tokenize(mem.text)
                freq_score = sum(self.word_frequency.get(w, 0) for w in words) / (len(words) + 1e-8)
                forget_bias = 0.5 if mem.level == 0 else 0.1
                if freq_score < Config.FORGET_THRESHOLD * forget_bias:
                    to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.memory.pop(i)
        if to_remove:
            logger.info(f"🧹 Иерархически забыто {len(to_remove)} элементов")

    def _perform_clustering(self):
        if len(self.memory) < 5:
            return
        vectors = np.array([m.vector for m in self.memory])
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(vectors)
        self.cluster_labels = clustering.labels_
        logger.info(f"Кластеризация: найдено {len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)} кластеров")

    async def learn(self, text: str, from_dialog: bool = False, user_feedback: str = "neutral") -> Dict:
        async with self.learn_lock:
            try:
                if self.sleeping:
                    return {"status": "sleeping", "response": "Zzz... Sin спит."}
                await self._update_dialog_context(text)
                words = await self.tokenize(text)
                if not words:
                    return {"status": "empty", "response": "Пустой ввод"}

                total_vec = np.zeros(self.embedder.dim)
                active_ids = []
                for word in words:
                    vec = await self.embedder.get_vector(word)
                    total_vec += vec
                    new_id = self.node_counter
                    node = Resonator(new_id, level=0)
                    node.pattern = vec.copy()
                    self.nodes[new_id] = node
                    node.excite(1.0)
                    active_ids.append(new_id)
                    self.node_counter += 1

                    self.memory.append(MemoryItem(
                        vector=vec.copy(),
                        text=word,
                        level=0,
                        timestamp=time.time(),
                        phase_cluster_id=self.assign_to_phase_cluster(node)
                    ))

                total_vec /= len(words)
                self.memory.append(MemoryItem(
                    vector=total_vec.copy(),
                    text=' '.join(words),
                    level=1,
                    timestamp=time.time()
                ))

                await self.hierarchical_forget()
                self._perform_clustering()
                self.request_count += 1

                # === ОБУЧЕНИЕ С ПОДКРЕПЛЕНИЕМ ===
                if user_feedback == "good":
                    self.rl_policy["ask_question"] += 0.1
                    self.rl_policy["generate"] += 0.1
                    logger.info("RL: Награда за хороший ответ")
                elif user_feedback == "bad":
                    self.rl_policy["ask_question"] -= 0.1
                    self.rl_policy["generate"] -= 0.1
                    logger.info("RL: Штраф за плохой ответ")

                return {"status": "learned", "response": f"Sin понял: '{text}'"}
            except Exception as e:
                self.error_count += 1
                logger.error(f"Ошибка в learn: {str(e)}")
                return {"status": "error", "response": "Ошибка при обучении."}

    async def respond(self, text: str) -> str:
        if self.pending_questions and random.random() < self.rl_policy["ask_question"]:
            return f"❓ {self.pending_questions.pop(0)}"
        if self.sleeping:
            return "Zzz... Sin спит."
        words = await self.tokenize(text)
        if not words:
            return "Я слушаю..."
        query_vec = np.mean([await self.embedder.get_vector(w) for w in words], axis=0)
        best_sim = 0.0
        best_match = None
        for mem in self.memory:
            try:
                sim = cosine_similarity([query_vec], [mem.vector])[0][0]
                if sim > best_sim:
                    best_sim = sim
                    best_match = mem.text
            except:
                continue
        if best_sim > 0.6:
            hints = ["Это напоминает мне о", "Я чувствую сходство с"]
            return f"{random.choice(hints)} '{best_match}' (схожесть: {best_sim:.2f})."
        elif best_sim > 0.4:
            return f"Частично понимаю. Ещё не до конца ясно ({best_sim:.2f})."
        else:
            return f"Новое. Ещё не резонирует. Расскажи больше."


# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ===
sin = Sin()


# === ВЕБ-ИНТЕРФЕЙС (Streamlit) ===
def run_web():
    st.set_page_config(page_title="Sin — Когнитивный агент", layout="wide")
    st.title("🧠 Sin v9.0 — Сеть Интуитивного Понимания")

    if st.button("Перезагрузить Sin"):
        asyncio.run(sin._init_system())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💬 Диалог")
        user_input = st.text_input("Ты:")
        if st.button("Отправить"):
            if user_input:
                asyncio.run(sin.learn(user_input))
                response = asyncio.run(sin.respond(user_input))
                st.session_state.chat.append(("Ты", user_input))
                st.session_state.chat.append(("Sin", response))
        for speaker, text in st.session_state.get('chat', []):
            st.write(f"**{speaker}**: {text}")

        feedback = st.radio("Оцените ответ:", ["neutral", "good", "bad"])
        if st.button("Отправить оценку"):
            asyncio.run(sin.learn("feedback", user_feedback=feedback))

    with col2:
        st.subheader("📊 Визуализация")
        if sin.activation_history:
            data = np.array(sin.activation_history)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(data.T, aspect='auto', cmap='plasma', interpolation='none')
            ax.set_title("Волны резонанса")
            st.pyplot(fig)

        if len(sin.cluster_labels) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(range(len(sin.cluster_labels)), sin.cluster_labels, c=sin.cluster_labels, cmap='tab10')
            ax.set_title("Кластеризация памяти")
            st.pyplot(fig)


# === API и Telegram — как в предыдущей версии ===
app = FastAPI(title=f"SIN API v{Sin.VERSION}")
semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await sin._init_system()
    yield

app.router.lifespan_context = lifespan

@app.post("/learn", dependencies=[Depends(lambda: semaphore.acquire())])
async def api_learn(request: LearnRequest):
    try:
        result = await sin.learn(request.text)
        return JSONResponse(result)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    finally:
        semaphore.release()

@app.post("/respond", dependencies=[Depends(lambda: semaphore.acquire())])
async def api_respond(request: RespondRequest):
    try:
        response = await sin.respond(request.text)
        return {"response": response}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    finally:
        semaphore.release()


# === TELEGRAM-БОТ ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Привет! Я SIN v{Sin.VERSION}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = await sin.respond(user_text)
    await update.message.reply_text(f"💬 Sin: {response}")

def run_telegram():
    try:
        app_bot = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        app_bot.add_handler(CommandHandler("start", start))
        app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app_bot.run_polling()
    except Exception as e:
        logger.critical(f"Ошибка Telegram-бота: {str(e)}")


# === ЗАПУСК ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "telegram", "web", "cli"], default="web")
    args = parser.parse_args()

    if args.mode == "web":
        st.session_state.chat = []
        run_web()
    elif args.mode == "api":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif args.mode == "telegram":
        run_telegram()
    else:
        # CLI
        print(f"🌀 SIN v{Sin.VERSION} — Консольный режим")
        while True:
            try:
                user_input = input("> Sin, ").strip()
                if user_input.lower() == "quit":
                    break
                print(f"💬 Sin: Это CLI, используйте --mode web для полного интерфейса")
            except KeyboardInterrupt:
                break
