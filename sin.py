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
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from gensim.models import KeyedVectors
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from contextlib import asynccontextmanager
import threading

# === КОНФИГУРАЦИЯ ===
class Config:
    # Пути
    EMBEDDING_PATH = r"C:\Users\alex\Downloads\cc.ru.300.vec"
    EMBEDDING_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
    PERSIST_FILE = "sin_state.pkl"
    LOG_FILE = "sin.log"

    # Параметры
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
    MAX_MEMORY_PERCENT = 80  # % RAM
    CACHE_SIZE = 1000

    # Telegram
    TELEGRAM_TOKEN = "7990254673:AAE-7UGlXLWnQ-Dn5D2uyrz0RYDJnBZZKM8"


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


# === Pydantic МОДЕЛИ ===
class LearnRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_TEXT_LENGTH)
    priority: int = Field(1, ge=1, le=5)

    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class RespondRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_TEXT_LENGTH)


# === АСИНХРОННЫЙ ХЭШ ===
async def async_get_md5(filepath: str) -> Optional[str]:
    """Асинхронное вычисление MD5"""
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
    """Асинхронная загрузка и распаковка"""
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

        # Распаковка
        logger.info("Распаковка .gz...")
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


# === КЭШ ДЛЯ ЧАСТО ИСПОЛЬЗУЕМЫХ ВЕКТОРОВ ===
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
                # Удаляем самый старый
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
            logger.info("🌀 Асинхронная загрузка эмбеддингов...")
            self.model = KeyedVectors.load_word2vec_format(self.filepath, binary=False, limit=300000)
            logger.info(f"✅ Загружено {len(self.model.key_to_index)} слов (dim={self.dim})")
            return True
        except Exception as e:
            logger.critical(f"Не удалось загрузить эмбеддинги: {str(e)}")
            return False

    async def get_vector(self, word: str) -> np.ndarray:
        try:
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
        except Exception as e:
            logger.error(f"Ошибка в get_vector: {e}")
            return np.zeros(self.dim)


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
    VERSION = "8.0"
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

            # Запуск фоновых задач
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

        # Проверка целостности
        expected_md5 = "d3e37867b88e742d386025b4d524515c"  # Пример
        file_md5 = await async_get_md5(Config.EMBEDDING_PATH)
        if file_md5 and file_md5.lower() != expected_md5.lower():
            logger.critical(f"MD5 не совпадает! Ожидалось: {expected_md5}, получено: {file_md5}")
            raise RuntimeError("Файл эмбеддингов повреждён или не тот.")

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
                        'phase_clusters': self.phase_clusters
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
            if memory > Config.MAX_MEMORY_PERCENT:
                logger.warning("Высокое потребление памяти! Запуск очистки...")
                await self._cleanup_memory()

    async def _cleanup_memory(self):
        # Простая очистка: удаляем самые старые элементы
        if len(self.memory) > 500:
            self.memory = self.memory[-300:]
            logger.info("Память очищена")

    async def tokenize(self, text: str) -> List[str]:
        try:
            words = [word.strip(".,!?\"'()[]{}:;—-") for word in text.lower().split() if word.isalpha()]
            for word in words:
                self.word_frequency[word] += 1
            return words
        except Exception as e:
            logger.error(f"Ошибка в tokenize: {e}")
            return []

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

    async def learn(self, text: str, from_dialog: bool = False) -> Dict:
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
                    # resonance = await self.check_resonance(vec)  # Реализуй async
                    resonance = 0.5  # временно
                    new_id = self.node_counter
                    node = Resonator(new_id, level=0)
                    node.pattern = vec.copy()
                    self.nodes[new_id] = node
                    node.excite(1.0)
                    active_ids.append(new_id)
                    self.node_counter += 1

                    if resonance < 0.6:
                        cluster_id = self.assign_to_phase_cluster(node)
                        self.memory.append(MemoryItem(
                            vector=vec.copy(),
                            text=word,
                            level=0,
                            timestamp=time.time(),
                            phase_cluster_id=cluster_id
                        ))

                total_vec /= len(words)
                self.memory.append(MemoryItem(
                    vector=total_vec.copy(),
                    text=' '.join(words),
                    level=1,
                    timestamp=time.time()
                ))

                await self.hierarchical_forget()
                self.request_count += 1
                return {"status": "learned", "response": f"Sin понял: '{text}'"}
            except Exception as e:
                self.error_count += 1
                logger.error(f"Ошибка в learn: {str(e)}")
                return {"status": "error", "response": "Ошибка при обучении."}


# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ===
sin = Sin()


# === API ===
semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
app = FastAPI(title=f"SIN API v{Sin.VERSION}")

async def rate_limit():
    await semaphore.acquire()

@app.post("/learn", dependencies=[Depends(rate_limit)])
async def api_learn(request: LearnRequest):
    try:
        result = await sin.learn(request.text)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"API /learn ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        semaphore.release()


# === TELEGRAM-БОТ ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Привет! Я SIN v{Sin.VERSION}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = "💬 Sin: Я ещё не умею отвечать в этом режиме, но учусь!"
    await update.message.reply_text(response)

def run_telegram():
    try:
        app_bot = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        app_bot.add_handler(CommandHandler("start", start))
        app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        logger.info("Telegram-бот запущен")
        app_bot.run_polling()
    except Exception as e:
        logger.critical(f"Ошибка Telegram-бота: {str(e)}")


# === CLI ===
def run_cli():
    print(f"🌀 SIN v{Sin.VERSION} — Консольный режим")
    while True:
        try:
            user_input = input("> Sin, ").strip()
            if user_input.lower() == "quit":
                break
            print(f"💬 Sin: Это CLI, ответ пока не реализован")
        except KeyboardInterrupt:
            break


# === ЗАПУСК ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    await sin._init_system()
    yield

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "telegram", "cli"], default="cli")
    args = parser.parse_args()

    if args.mode == "api":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif args.mode == "telegram":
        run_telegram()
    else:
        run_cli()
